"""Reference (oracle) implementations for model paths `tests/reference.py`
does not reach, written from the documented objective and semantics rather
than ported from the core.

The models whose state is a sum -- :func:`lasso_paths_ref`,
:func:`ewridge_paths_ref`, :func:`rls_paths_ref` -- are recomputed **from the
raw rows** at the moment each statistic is needed, where
`tests/reference.py` runs each model's own recursion longhand: each past row
at its weight ``w * 0.5 ** (age / half_life)``, the age read on the capped
clock the decay uses. That makes them independent of the mean-form
recursion as well as of the solver, and it is what lets a ``window`` be
written from its definition -- the rows whose age is less than ``window``
(at most it under ``closed="both"``), nothing else -- instead of from the
snapshot subtraction the core uses to get there. The models that keep no
sums -- :func:`pa_ref`, :func:`sgd_ref`, :func:`holt_ref` -- are their
builders' update equations, written out.

Conventions shared with the core (docs/PLAN.md sections 3 and 13, CLAUDE.md
hard rules 2, 8 and 9):

- A row with a null feature is skipped: every output NaN, nothing learned,
  and its clock step folds into the next accepted row's, which
  ``gap_cap`` caps.
- ``weight_sum`` is the weight before the row's update and before its own decay,
  so its ages are counted from the last accepted row; under a ``window`` it
  is the weight inside the window, as seen from there.
- A zero-weight row advances the clock and teaches nothing.
"""

from __future__ import annotations

import numpy as np

from reference import _enet_descent


def _within(horizon: float, closed: str):
    """The rows a window of ``horizon`` holds, by their ages: less than it
    under ``closed=\"right\"``, the default, at most it under ``\"both\"``
    (docs/PLAN.md task 196)."""
    if closed == "right":
        return lambda ages: ages < horizon
    return lambda ages: ages <= horizon


def _weighted_mean(v: np.ndarray, om: np.ndarray) -> np.ndarray:
    return (om[:, None] * v).sum(axis=0) / om.sum()


def _held(x: np.ndarray, om: np.ndarray) -> np.ndarray:
    """The columns that hold one value on every row of positive weight: no
    spread at all, which is what a variance of exactly zero means. Tested on
    the values, not on a variance recomputed from them, where the weighted
    mean of equal values can miss them by an ulp and leave a variance of
    ulp squared (docs/PLAN.md task 94)."""
    rows = x[om > 0.0]
    if rows.shape[0] == 0:
        return np.zeros(x.shape[1], dtype=bool)
    return (rows == rows[0]).all(axis=0)


def lasso_paths_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    lasso_path: list[float],
    *,
    l1_ratio: float = 1.0,
    half_life: float = np.inf,
    select_half_life: float | None = None,
    min_weight: float | list[float] | None = None,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    gap_cap: float = np.inf,
    window_size: float | None = None,
    closed: str = "right",
    fit_intercept: bool = True,
    target_gaps: str = "own_rows",
    tol: float = 1e-14,
) -> dict[str, np.ndarray]:
    """The ``lasso`` builder's path over several targets, null targets,
    ``target_gaps``, ``window_size``, ``fit_intercept=False`` and
    ``lam_selected`` (docs/PLAN.md section 4.3 and task 81, the ``lasso`` and
    ``ewridge`` docstrings).

    **The fit.** Each solve fits every path point ``l`` of every target
    ``j`` from the rows learned so far, each at ``w * 0.5 ** (age /
    half-life)`` with the age counted from the solve's row, minimising::

        1/2 b'Cb - c'b + l1 |b|_1 + l2/2 |b|^2      l1 = l * l1_ratio, l2 = l * (1 - l1_ratio)

    by :func:`reference._enet_descent`, from zero to ``tol``. Which rows:

    - ``"own_rows"``: ``C``, ``c``, the means and the scales over the rows the
      target is present on -- the fit of the frame with its nulls dropped;
    - ``"pairwise"``: ``C`` and the scales over every row; ``c`` over the
      target's rows, centred at the target's own means (pandas'
      pairwise-complete covariance). The intercept reads the target's own
      means either way;
    - ``window_size``: of those, only the rows whose age is less than
      ``window_size``, or at most it under ``closed="both"``.

    With an intercept, ``C`` is the correlation matrix of the centred
    features, ``c_i = cov(x_i, y) / s_i``, ``coef_i = b_i / s_i`` and the
    intercept is ``ybar - m . coef``; a feature whose variance is zero is
    dropped with coefficient 0 (the ``ewridge`` docstring; the ``1e-10``
    threshold ``lasso_ref`` cites went with T-E9), which is to say one that
    holds a single value on every row of the Gram (``_held``). Without one
    nothing is centred: ``C = E[x x'] / (r r')``, ``c_i = E[x_i y] / r_i`` with ``r_i``
    the root mean square, ``coef_i = b_i / r_i``, and a column that is all
    zero is dropped.

    A problem is held only where the smallest eigenvalue of ``C + l2 I`` is
    at least 1e-2, and a target with no weight has no problem to hold; either
    is NaN here, and a row scored with one raises (``lasso_ref``'s rule).
    Under a ``window`` a feature can hold one value on every row inside it --
    one row left, say -- and is dropped there. The core rebuilds a window's
    Gram by subtracting the state at its boundary, which left such a
    variance as rounding noise that it kept, so that the zero-penalty slope
    was noise over noise; each feature's run of equal values now says the
    variance is zero exactly (docs/PLAN.md task 94), and these problems are
    held.

    **Scoring.** Row ``i`` is scored with the last solve before it, target
    ``j`` only once its own weight -- the rows it is present on, at their raw
    weights, decayed, inside the window -- reaches its ``min_weight``
    (scalar or one per target). ``weight_sum`` is the weight of every row.

    **The schedule** is ``lasso_ref``'s: after the row is learned a solve
    runs when the clock since the last one reaches ``solve_every`` (left out:
    once the weight learned since the last reaches ``ln 2 / 50`` of the weight
    the fit holds under a finite half-life, docs/PLAN.md task 115 (b); every row
    for an infinite one), when
    ``max_rows_between_solves`` rows have gone by, or when there has been
    none yet and ``weight_sum`` has reached ``min_weight``. That last rule is
    written for one target and one threshold; with several the reference
    raises if it is ever the one that decides, so a test must let the cadence
    solve first.

    **``lam_selected``.** Per target, the path point with the least weighted
    sum of squared out-of-sample errors so far, each at ``w * 0.5 ** (age /
    select_half_life)`` (default the half-life) and, under a window, inside it
    -- as it stood before the row. The errors are those of the predictions
    ``pred`` shows, on the rows the target is present: from the row where the
    target's own weight reaches its own ``min_weight``, so a row its threshold
    withholds adds none, whatever the other targets' thresholds (review
    2026-10-05, CA3). NaN where there is no error yet, and where the best two
    are within 1e-9 of each other (a tie is not something to hold a library
    to).

    Returns ``pred`` (n, m, P), ``weight_sum`` (n,), ``w_target`` (n, m), each
    target's weight before the row, ``coef`` (n, m, P, kt), the last solve's
    fit, NaN before the first and on skipped rows, ``lam_selected`` (n, m)
    and ``solved`` (n,).
    """
    n, k = X.shape
    m = Y.shape[1]
    kt = k + 1 if fit_intercept else k
    npath = len(lasso_path)
    path = np.asarray(lasso_path, dtype=float)
    if target_gaps not in ("own_rows", "pairwise"):
        raise ValueError(target_gaps)
    if min_weight is None:
        min_weight = float(kt)
    mp = np.broadcast_to(np.asarray(min_weight, dtype=float), (m,))
    share = np.log(2.0) / 50.0 if solve_every is None and np.isfinite(half_life) else None
    if solve_every is None:
        solve_every = 0.0
    since_w = 0.0
    if select_half_life is None:
        select_half_life = half_life
    max_rows = np.inf if max_rows_between_solves is None else max_rows_between_solves
    horizon = np.inf if window_size is None else window_size
    within = _within(horizon, closed)
    single = m == 1 and np.ndim(min_weight) == 0

    pred = np.full((n, m, npath), np.nan)
    weight_sum = np.full(n, np.nan)
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
        if fit_intercept:
            mg = _weighted_mean(xg, omg)
            cov = (omg[:, None, None] * np.einsum("ri,rj->rij", xg - mg, xg - mg)).sum(
                axis=0
            ) / omg.sum()
            mo = _weighted_mean(xo, omo)
            ybar = float(omo @ yo / omo.sum())
            cxy = ((omo * (yo - ybar))[:, None] * (xo - mo)).sum(axis=0) / omo.sum()
            var = np.diag(cov)
            kept = np.flatnonzero((var > 0.0) & ~_held(xg, omg))
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
        floor = np.linalg.eigvalsh(C)[0] if kept.size else np.inf
        off = 1 if fit_intercept else 0
        for p, lam in enumerate(path):
            l1, l2 = lam * l1_ratio, lam * (1.0 - l1_ratio)
            if floor + l2 < 1e-2:
                continue
            b = np.zeros(k)
            b[kept] = _enet_descent(C, c, l1, l2, tol) / s
            fit[p, off:] = b
            if fit_intercept:
                fit[p, 0] = ybar - mo @ b
        return fit

    def solve(t_now: float) -> np.ndarray:
        ta, wa, xa, ya = (np.asarray(v) for v in (T, Wr, Xr, Yr))
        ages = t_now - ta
        inside = within(ages)
        om = wa * decay(ages, half_life)
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

    fit = None
    t_last, pending = 0.0, 0.0
    since_clock, since_rows, since_w = 0.0, 0, 0.0
    for i in range(n):
        if np.isnan(X[i]).any():
            pending += dclock[i]
            continue
        d = min(dclock[i] + pending, gap_cap)
        pending = 0.0
        t_now = (t_last + d) if T else 0.0
        z = np.concatenate(([1.0], X[i])) if fit_intercept else X[i]

        # ---- before the row: every weight is seen from the last accepted row ----
        if T:
            ages = t_last - np.asarray(T)
            inside = within(ages)
            om = np.where(inside, np.asarray(Wr) * decay(ages, half_life), 0.0)
            present = ~np.isnan(np.asarray(Yr))
            weight_sum[i] = om.sum()
            w_target[i] = om @ present
        else:
            weight_sum[i] = 0.0
            w_target[i] = 0.0
        for j in range(m):
            # The model predicts the target once ``weight_sum`` reaches the
            # smallest threshold and the target has weight; the output shows
            # it once its own weight reaches its own.
            model = fit is not None and w_target[i, j] > 0.0 and weight_sum[i] >= float(np.min(mp))
            if model and w_target[i, j] >= mp[j]:
                p = fit[j] @ z
                if np.isnan(p).any():
                    raise ValueError(
                        f"row {i}, target {j} is scored with a fit the reference does not hold"
                    )
                pred[i, j] = p
            if errs[j]:
                te, we, ee = (np.asarray(v) for v in zip(*errs[j], strict=True))
                ages_e = t_last - te
                wt = np.where(within(ages_e), we * decay(ages_e, select_half_life), 0.0)
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
            if np.isnan(pred[i, j]).any() or np.isnan(Y[i, j]) or not w[i] > 0.0:
                continue
            errs[j].append((t_now, float(w[i]), Y[i, j] - pred[i, j]))
        t_last = t_now

        # ---- solve, on the schedule ----
        # A zero-weight row is clock alone (hard rule 9): it never solves and
        # is no row of the cap, and a solve due on it waits for the next row
        # with weight (docs/PLAN.md task 214).
        since_clock += d
        teaches = float(w[i]) > 0.0
        since_rows += int(teaches)
        since_w += max(float(w[i]), 0.0)
        if share is not None:
            ages = t_now - np.asarray(T)
            held = (np.asarray(Wr) * decay(ages, half_life))[within(ages)].sum()
            by_cadence = since_w >= share * held
        else:
            by_cadence = solve_every <= 0.0 or since_clock >= solve_every
        cadence = teaches and (by_cadence or since_rows >= max_rows)
        if teaches and not cadence and fit is None:
            ages = t_now - np.asarray(T)
            weight = (np.asarray(Wr) * decay(ages, half_life))[within(ages)].sum()
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
            since_clock, since_rows, since_w = 0.0, 0, 0.0
        if fit is not None:
            coef[i] = fit

    return {
        "pred": pred,
        "weight_sum": weight_sum,
        "w_target": w_target,
        "coef": coef,
        "lam_selected": lam_selected,
        "solved": solved,
    }


def _share(kf: np.ndarray, ks: np.ndarray, rows: np.ndarray, f: float) -> np.ndarray:
    """A ``session_shrink`` blend of one accumulator over ``rows``: ``1 - f`` of
    today's kernel ``kf`` and ``f`` of the twin's ``ks``, each normalised by
    its weight there, at today's weight (docs/PLAN.md task 145). Either side
    with no weight there leaves it as it is, as the model's blend does."""
    wf, ws = float(kf[rows].sum()), float(ks[rows].sum())
    if not (wf > 0.0 and ws > 0.0):
        return kf
    out = kf.copy()
    out[rows] = wf * ((1.0 - f) * kf[rows] / wf + f * ks[rows] / ws)
    return out


def _ridge_fit(xg, omg, xo, yo, omo, ridge, standardize, fit_intercept):
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

    A feature that holds one value on every row of the Gram (``_held``) has
    a variance and a cross-moment of zero, so the ridge gives it a slope of
    zero and ``standardize`` drops it; under a window that is one row left,
    say (``lasso_paths_ref`` says why this is held now).

    Returns ``(intercept, slopes)``, or ``None`` where the reference does not
    hold the problem: a target with no weight, or a feature set whose
    features are nearly collinear on these rows (the smallest eigenvalue of
    their correlation matrix below 1e-3)."""
    if omo.sum() <= 0.0 or omg.sum() <= 0.0:
        return None
    if fit_intercept:
        mg = _weighted_mean(xg, omg)
        M = (omg[:, None, None] * np.einsum("ri,rj->rij", xg - mg, xg - mg)).sum(axis=0) / omg.sum()
        mo = _weighted_mean(xo, omo)
        ybar = float(omo @ yo / omo.sum())
        v = ((omo * (yo - ybar))[:, None] * (xo - mo)).sum(axis=0) / omo.sum()
        # A feature holding one value on every row of the Gram has no spread
        # and no covariance with anything there, the target's rows included.
        held = _held(xg, omg)
        M[held, :] = 0.0
        M[:, held] = 0.0
        v[held] = 0.0
    else:
        M = (omg[:, None, None] * np.einsum("ri,rj->rij", xg, xg)).sum(axis=0) / omg.sum()
        v = ((omo * yo)[:, None] * xo).sum(axis=0) / omo.sum()
    scale = np.sqrt(np.diag(M))
    kept = np.flatnonzero(scale > 0.0)
    corr = M[np.ix_(kept, kept)] / np.outer(scale[kept], scale[kept])
    if kept.size and np.linalg.eigvalsh(corr)[0] < 1e-3:
        return None
    b = np.zeros(len(scale))
    if standardize:
        b[kept] = np.linalg.solve(corr + ridge * np.eye(kept.size), v[kept] / scale[kept])
        b[kept] /= scale[kept]
    else:
        b = np.linalg.solve(M + ridge * np.eye(len(scale)), v)
    return (ybar - mo @ b if fit_intercept else 0.0), b


def ewridge_paths_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    *,
    half_life: float,
    ridge: float | list[float] = 1e-6,
    feature_sets: list[list[int]] | None = None,
    standardize: bool = False,
    fit_intercept: bool = True,
    target_gaps: str = "own_rows",
    window_size: float | None = None,
    closed: str = "right",
    min_weight: float | list[float] | None = None,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    gap_cap: float = np.inf,
    session: np.ndarray | None = None,
    session_gap: float | str | None = None,
    session_shrink: float | None = None,
    long_half_life: float | None = None,
) -> dict[str, object]:
    """``ewridge`` on its documented schedule, over a ridge grid and feature
    sets, from the raw rows (the ``ewridge`` builder's docstring, docs/PLAN.md
    section 4.1 and task 81).

    Every past row of the current state enters a solve at its **effective
    weight**: ``w * 0.5 ** (age / half_life)``, with the age on the capped
    clock, inside the window under one. A ``session_shrink`` blend at ``f``
    fits on ``1 - f`` of today's rows and ``f`` of the long run's, at today's
    weight (docs/PLAN.md task 145): each row's weight becomes ``W_h · ((1 -
    f) ω_h / W_h + f ω_H / W_H)``, ``ω_h`` its weight here, ``ω_H`` its
    weight in the slow twin at ``long_half_life``, and ``W`` each side's total,
    as the rows stand before the first row of the new session. The Gram
    normalises over every row, and each target over its own rows, so a row
    keeps one weight per target beside the Gram's; the mean-form sums are
    linear in these weights, so the blend is exact on them
    (``tests/test_second_opinion.py`` ``TestSessionShrinkBlend`` pins that
    reading). The new session's first row is then scored from the blend,
    re-solved, and every row ages from there at ``half_life``.
    ``session_gap="reset"`` starts the state over.

    **Scoring.** Target ``j`` is scored with the last solve once its own
    weight (the rows it is present on) reaches its ``min_weight``, scalar or
    one per target. Left out, it is the builder's rule: the model's own
    floor, a row per coefficient (the features and the intercept), on
    ``weight_sum``, and no floor of a target's own beyond a weight above 0
    (``stream.rs::build_one``; ``spec.rs::default_min_periods`` is 0 for
    ``ewridge``). ``weight_sum`` is the weight of every row. Both are seen
    from the last accepted row, before the row's own decay (hard rule 8).

    **The schedule** is ``lasso_paths_ref``'s: after the row is learned a
    solve runs when the clock since the last one reaches ``solve_every``
    (left out: once the weight learned since the last reaches ``ln 2 / 50`` of
    the weight the fit holds under a finite half-life, docs/PLAN.md task 115
    (b); every row for ``inf``), when
    ``max_rows_between_solves`` rows have gone by, or when there has been
    none yet and ``weight_sum`` has reached the smallest ``min_weight`` -- by
    default the coefficient count.

    **The combinations** are every feature set (column indices; ``None`` is
    all of them) crossed with every ridge value, set-major. Each is solved by
    :func:`_ridge_fit` into a full-length coefficient vector, zero outside its
    set.

    Returns ``pred`` (n, m, C), ``weight_sum`` (n,), ``coef`` (n, m, C, kt) --
    NaN before the first solve, on a skipped row, and for a problem the
    reference does not hold, which a scored row may not use -- ``solved``
    (n,), ``has_fit`` (n,), the rows after which the model has a solve (a
    reset takes it away), and ``combos``, the (feature set, ridge) of each
    C.
    """
    n, k = X.shape
    m = Y.shape[1]
    kt = k + 1 if fit_intercept else k
    ridges = [ridge] if np.isscalar(ridge) else list(ridge)
    sets = [list(range(k))] if feature_sets is None else [list(s) for s in feature_sets]
    combos = [(s, r) for s in sets for r in ridges]
    if min_weight is None:
        floor, mp = float(kt), np.zeros(m)
    else:
        mp = np.broadcast_to(np.asarray(min_weight, dtype=float), (m,))
        floor = float(np.min(mp))
    share = np.log(2.0) / 50.0 if solve_every is None and np.isfinite(half_life) else None
    if solve_every is None:
        solve_every = 0.0
    since_w = 0.0
    max_rows = np.inf if max_rows_between_solves is None else max_rows_between_solves
    horizon = np.inf if window_size is None else window_size
    within = _within(horizon, closed)
    blend = session_shrink is not None
    if blend and long_half_life is None:
        raise ValueError("session_shrink needs long_half_life")

    pred = np.full((n, m, len(combos)), np.nan)
    weight_sum = np.full(n, np.nan)
    coef = np.full((n, m, len(combos), kt), np.nan)
    solved = np.zeros(n, dtype=bool)
    has_fit = np.zeros(n, dtype=bool)

    def factor(d: float, h: float) -> float:
        return 1.0 if np.isinf(h) else 0.5 ** (d / h)

    def solve() -> np.ndarray:
        wa, xa, ya, ta = np.asarray(fast), np.asarray(Xr), np.asarray(Yr), np.asarray(T)
        wt = np.asarray(fast_t)
        inside = within(t_last - ta)
        fit = np.full((m, len(combos), kt), np.nan)
        for j in range(m):
            own = inside & ~np.isnan(ya[:, j])
            gram = own if target_gaps == "own_rows" else inside
            wg = wt[:, j] if target_gaps == "own_rows" else wa
            for c, (cols, r) in enumerate(combos):
                got = _ridge_fit(
                    xa[gram][:, cols],
                    wg[gram],
                    xa[own][:, cols],
                    ya[own, j],
                    wt[own, j],
                    r,
                    standardize,
                    fit_intercept,
                )
                if got is None:
                    continue
                fit[j, c] = 0.0
                if fit_intercept:
                    fit[j, c, 0] = got[0]
                fit[j, c, (1 if fit_intercept else 0) + np.asarray(cols)] = got[1]
        return fit

    def restart():
        return [], [], [], [], [], None, 0.0, 0

    fast, slow, T, Xr, Yr, fit, since_clock, since_rows = restart()
    # Each row's weight in each target's own accumulator: the Gram's, until a
    # blend normalises each over its own rows.
    fast_t: list[np.ndarray] = []

    since_w = 0.0
    t_last, pending, prev_session, started = 0.0, 0.0, None, False
    for i in range(n):
        if np.isnan(X[i]).any():
            pending += dclock[i]
            continue
        changed = session is not None and started and session[i] != prev_session
        prev_session = None if session is None else session[i]
        if changed and session_gap == "reset":
            fast, slow, T, Xr, Yr, fit, since_clock, since_rows = restart()
            fast_t = []
            since_w = 0.0
            d = 0.0
        elif changed and session_gap is not None:
            d = min(float(session_gap), gap_cap)
        else:
            d = min(max(dclock[i], 0.0) + pending, gap_cap) if started else 0.0
        pending = 0.0
        started = True
        if changed and blend and fast:
            ks = np.asarray(slow)
            every = np.ones(len(ks), dtype=bool)
            fast = list(_share(np.asarray(fast), ks, every, session_shrink))
            wt = np.asarray(fast_t)
            present = ~np.isnan(np.asarray(Yr))
            for j in range(m):
                wt[:, j] = _share(wt[:, j], ks, present[:, j], session_shrink)
            fast_t = list(wt)
            fit = solve()
            since_clock, since_rows, since_w = 0.0, 0, 0.0
        z = np.concatenate(([1.0], X[i])) if fit_intercept else X[i]

        # ---- before the row, seen from the last accepted row ----
        if fast:
            inside = within(t_last - np.asarray(T))
            om = np.where(inside, np.asarray(fast), 0.0)
            weight_sum[i] = om.sum()
            wt = np.where(inside[:, None], np.asarray(fast_t), 0.0)
            w_target = (wt * ~np.isnan(np.asarray(Yr))).sum(axis=0)
        else:
            weight_sum[i], w_target = 0.0, np.zeros(m)
        for j in range(m):
            gated = w_target[j] >= mp[j] and weight_sum[i] >= floor
            if fit is not None and w_target[j] > 0.0 and gated:
                if np.isnan(fit[j]).any():
                    raise ValueError(f"row {i}, target {j} is scored with an unheld fit")
                pred[i, j] = fit[j] @ z

        # ---- learn: age every row by this row's step, then take it ----
        lam = factor(d, half_life)
        fast = [v * lam for v in fast] + [float(w[i])]
        fast_t = [v * lam for v in fast_t] + [np.full(m, float(w[i]))]
        if blend:
            lam_s = factor(d, long_half_life)
            slow = [v * lam_s for v in slow] + [float(w[i])]
        t_last = (t_last + d) if T else 0.0
        T.append(t_last)
        Xr.append(X[i].copy())
        Yr.append(Y[i].copy())

        # ---- solve, on the schedule ----
        # A zero-weight row is clock alone (hard rule 9): it never solves and
        # is no row of the cap, and a solve due on it waits for the next row
        # with weight (docs/PLAN.md task 214).
        since_clock += d
        teaches = float(w[i]) > 0.0
        since_rows += int(teaches)
        since_w += max(float(w[i]), 0.0)
        inside = within(t_last - np.asarray(T))
        if share is not None:
            by_cadence = since_w >= share * float(np.asarray(fast)[inside].sum())
        else:
            by_cadence = solve_every <= 0.0 or since_clock >= solve_every
        due = teaches and (by_cadence or since_rows >= max_rows)
        if teaches and not due and fit is None:
            due = float(np.asarray(fast)[inside].sum()) >= floor
        if due:
            fit = solve()
            solved[i] = True
            since_clock, since_rows, since_w = 0.0, 0, 0.0
        if fit is not None:
            coef[i] = fit
            has_fit[i] = True

    return {
        "pred": pred,
        "weight_sum": weight_sum,
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
    half_life: float,
    delta: float = 1.0,
    coef_prior: np.ndarray | None = None,
    fit_intercept: bool = True,
    min_weight: float | None = None,
    gap_cap: float = np.inf,
) -> dict[str, np.ndarray]:
    """``rls`` from its documented recursion (the ``rls`` builder's
    docstring), summed from the raw rows rather than rotated in::

        A <- lam * A + w * z z'        b_j <- lam * b_j + w * y_j * z        beta_j = A^-1 b_j
        A_0 = delta * I                b_0 = delta * coef_prior

    so after rows at ages ``a_r`` (the prior's age is the whole stream's),
    ``A = delta * lam^T I + sum_r w_r lam^a_r z_r z_r'`` and ``b_j`` likewise.
    The factor ``R`` is shared, so a row with any null target is learned
    for none (it still ages the sums). Scoring: once the weight of the rows
    learned from reaches ``min_weight`` (default the number of
    coefficients; hard rule 8, docs/PLAN.md task 115 (d)); the prior alone
    predicts nothing (review 2026-09-12, S10). ``weight_sum`` is the weight of
    every row. ``coef`` is ``beta`` after the row.

    Returns ``pred`` and ``coef`` per target, and ``weight_sum``."""
    n, k = X.shape
    m = Y.shape[1]
    kt = k + 1 if fit_intercept else k
    if min_weight is None:
        min_weight = float(kt)
    prior = np.zeros((m, kt)) if coef_prior is None else np.asarray(coef_prior, float)
    pred = np.full((n, m), np.nan)
    weight_sum = np.full(n, np.nan)
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
        om = np.asarray(Wr) * 0.5 ** (age / half_life) * np.asarray(learned)
        prior_w = delta * 0.5 ** (t / half_life)
        za, ya = np.asarray(Z), np.nan_to_num(np.asarray(Yr))
        A = prior_w * np.eye(kt) + (om[:, None, None] * np.einsum("ri,rj->rij", za, za)).sum(axis=0)
        b = prior_w * prior.T + (om[:, None, None] * np.einsum("ri,rj->rij", za, ya)).sum(axis=0)
        return np.linalg.solve(A, b).T

    for i in range(n):
        if np.isnan(X[i]).any():
            pending += dclock[i]
            continue
        d = min(dclock[i] + pending, gap_cap) if started else 0.0
        pending = 0.0
        if started:
            aged = np.asarray(Wr) * 0.5 ** ((t_now - np.asarray(T)) / half_life)
            weight_sum[i] = float(aged.sum())
            w_learned = float((aged * np.asarray(learned)).sum())
        else:
            weight_sum[i], w_learned = 0.0, 0.0
        z = np.concatenate(([1.0], X[i])) if fit_intercept else X[i].copy()
        if beta is not None and any(learned) and w_learned >= min_weight:
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
    return {"pred": pred, "weight_sum": weight_sum, "coef": coef}


def _accepted_steps(X: np.ndarray, dclock: np.ndarray, gap_cap: float):
    """Yield ``(i, d)`` for each accepted row: a row with a null feature is
    skipped, its clock step folding into the next accepted row's, which
    ``gap_cap`` caps; the first accepted row's step is 0."""
    pending, started = 0.0, False
    for i in range(X.shape[0]):
        if np.isnan(X[i]).any():
            pending += dclock[i]
            continue
        d = min(dclock[i] + pending, gap_cap) if started else 0.0
        pending, started = 0.0, True
        yield i, d


def _ew_variance(history: list[list[float]]) -> float:
    """The EW variance of a target around its EW mean, from its definition:
    ``sum(v * (y - m) ** 2) / sum(v)`` with ``m = sum(v * y) / sum(v)``, over
    the ``[y, v]`` pairs of the rows that carried it, ``v`` each row's weight
    aged since. 0 before two rows."""
    if len(history) < 2:
        return 0.0
    y = np.array([h[0] for h in history])
    v = np.array([h[1] for h in history])
    mean = float(np.sum(v * y) / np.sum(v))
    return float(np.sum(v * (y - mean) ** 2) / np.sum(v))


def _age_and_learn(history: list[list[float]], lam: float, y: float, w: float) -> None:
    """One row of a target's spread: every weight aged by ``lam``, and the
    row's ``y`` joining at its weight where it is present with one above 0."""
    for h in history:
        h[1] *= lam
    if not np.isnan(y) and w > 0.0:
        history.append([y, w])


#: Kish's count of the rows a standardizing model's scaler learns before its
#: fit is held in the caller's units (the ``sgd`` builder's ``standardize``
#: docstring; docs/PLAN.md task 206).
WARMUP_ROWS = 22.0


def _moments(
    rows: list[np.ndarray], om: list[float], fit_intercept: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The scaler's means and scales by their definition, over the rows
    ``rows`` at the aged weights ``om``: ``m_i`` the EW mean and ``s_i`` the
    EW standard deviation, 1 where there is no spread; without an intercept
    ``m_i = 0`` and ``s_i`` the root of the raw second moment, 1 where it is
    0. Two passes over the rows; a feature every weighted row holds at one
    value has no spread, exactly, as the core's pairs give it."""
    k = len(rows[0]) if rows else 0
    m, s = np.zeros(k), np.ones(k)
    if not rows:
        return m, s
    X, v = np.vstack(rows), np.asarray(om)
    if not np.sum(v) > 0.0:
        return m, s
    for i in range(k):
        col = X[:, i]
        if fit_intercept:
            held = col[v > 0.0]
            same = bool(np.all(held == held[0]))
            mean = float(held[0]) if same else float(np.sum(v * col) / np.sum(v))
            var = 0.0 if same else float(np.sum(v * (col - mean) ** 2) / np.sum(v))
            m[i] = mean
            s[i] = np.sqrt(var) if var > 0.0 else 1.0
        else:
            raw = float(np.sum(v * col**2) / np.sum(v))
            s[i] = np.sqrt(raw) if raw > 0.0 else 1.0
    return m, s


def _row_map(
    rows: list[np.ndarray], om: list[float], x: np.ndarray, fit_intercept: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The row's map under ``standardize`` (the ``sgd`` builder's docstring):
    the moments with the row itself admitted at unit weight, ``rows`` and
    ``om`` the earlier rows already aged by this row's decay. ``z``, ``m``
    and ``s`` over the coefficient slots, the intercept's ``(1, 0, 1)``
    first: ``z_i = (x_i - m_i) / s_i``, or ``x_i / s_i`` without an
    intercept."""
    m, s = _moments([*rows, x], [*om, 1.0], fit_intercept)
    z = (x - m) / s
    if fit_intercept:
        return (
            np.concatenate(([1.0], z)),
            np.concatenate(([0.0], m)),
            np.concatenate(([1.0], s)),
        )
    return z, m, s


def _read_out(beta: np.ndarray, m: np.ndarray, s: np.ndarray, fit_intercept: bool) -> np.ndarray:
    """Coefficients in the scaler's coordinates read in the caller's units
    with the moments ``m``, ``s`` as they stand: ``b_i = beta_i / s_i`` and
    ``b_0 = beta_0 - sum_i b_i m_i``."""
    off = 1 if fit_intercept else 0
    b = beta.copy()
    b[off:] = beta[off:] / s
    if fit_intercept:
        b[0] = beta[0] - float(np.sum(b[1:] * m))
    return b


def _mapped(dbeta: np.ndarray, m: np.ndarray, s: np.ndarray, fit_intercept: bool) -> np.ndarray:
    """A step taken in the row's standardized coordinates, in the caller's
    units (``m``, ``s`` over the coefficient slots, the intercept's 0 and 1
    in): ``db_i = dbeta_i / s_i``, and ``db_0 = dbeta_0 - sum_i m_i db_i``."""
    db = dbeta / s
    if fit_intercept:
        db[0] = dbeta[0] - float(np.sum(m[1:] * db[1:]))
    return db


def _box(
    k: int, lo: float | None, hi: float | None, total: float | None
) -> tuple[np.ndarray, np.ndarray, float | None] | None:
    """The bounds a constraint gives, one per slope, or ``None`` where none
    is given."""
    if lo is None and hi is None and total is None:
        return None
    return (
        np.full(k, -np.inf if lo is None else lo),
        np.full(k, np.inf if hi is None else hi),
        total,
    )


def _nearest(
    v: np.ndarray,
    step: np.ndarray,
    weight: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    total: float | None,
) -> np.ndarray:
    """``clip(v - mu * step, lo, hi)`` for the one ``mu`` at which ``sum(weight
    * x) = total`` (0 without a sum): the nearest point of the box and the
    sum in the metric ``sum((x_i - v_i) ** 2 / (step_i / weight_i))``, from
    its KKT conditions. ``mu`` is found by bisection on the sum, which falls
    as ``mu`` grows, not by the breakpoints the core sorts."""

    def at(mu: float) -> np.ndarray:
        return np.clip(v - mu * step, lo, hi)

    if total is None:
        return at(0.0)
    lo_mu, hi_mu = -1.0, 1.0
    while float(np.sum(weight * at(lo_mu))) < total:
        lo_mu *= 2.0
    while float(np.sum(weight * at(hi_mu))) > total:
        hi_mu *= 2.0
    for _ in range(400):
        mid = 0.5 * (lo_mu + hi_mu)
        if mid in (lo_mu, hi_mu):
            break
        if float(np.sum(weight * at(mid))) > total:
            lo_mu = mid
        else:
            hi_mu = mid
    return at(0.5 * (lo_mu + hi_mu))


def _projected_standardized(
    beta: np.ndarray, box: tuple, s: np.ndarray, fit_intercept: bool
) -> np.ndarray:
    """While the scaler warms up: the slopes ``beta_i`` in the scaler's
    coordinates, ``beta_i = s_i b_i``, projected Euclidean there, the
    caller's bounds carried over, ``lo_i s_i <= beta_i <= hi_i s_i`` and
    ``sum(beta_i / s_i) = total``; the intercept free (the ``sgd`` builder's
    ``coef_min`` docstring)."""
    lo, hi, total = box
    off = 1 if fit_intercept else 0
    out = beta.copy()
    out[off:] = _nearest(beta[off:], 1.0 / s, 1.0 / s, lo * s, hi * s, total)
    return out


def _projected_in_units(
    b: np.ndarray, box: tuple, m: np.ndarray, s: np.ndarray, fit_intercept: bool
) -> np.ndarray:
    """Past the warm-up: the slopes in the caller's units projected on the
    caller's box and sum in the metric of the row's standardized
    coordinates, ``b_i = clip(b_i - mu / s_i ** 2, lo_i, hi_i)``, and the
    intercept keeping its standardized value ``b_0 + sum_i m_i b_i`` (``m``,
    ``s`` over the coefficient slots)."""
    lo, hi, total = box
    off = 1 if fit_intercept else 0
    out = b.copy()
    sl = s[off:]
    out[off:] = _nearest(b[off:], 1.0 / sl**2, np.ones_like(sl), lo, hi, total)
    if fit_intercept:
        out[0] += float(np.sum(m[1:] * (b[1:] - out[1:])))
    return out


def _decay(half_life: float, decay_lam: float | None, d: float) -> float:
    """The decay over a clock step ``d``: ``decay_lam ** d`` where a factor per
    clock unit is given, else ``0.5 ** (d / half_life)``."""
    if decay_lam is not None:
        return decay_lam**d
    return 1.0 if np.isinf(half_life) else 0.5 ** (d / half_life)


def pa_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    *,
    mode: str = "pa1",
    c: float = 1.0,
    eps: float = 0.01,
    half_life: float = np.inf,
    fit_intercept: bool = True,
    min_weight: float = 0.0,
    gap_cap: float = np.inf,
    standardize: bool = False,
    decay_lam: float | None = None,
    coef_min: float | None = None,
    coef_max: float | None = None,
    coef_sum: float | None = None,
) -> dict[str, np.ndarray]:
    """Passive-aggressive regression, Crammer et al. (2006), as the ``pa``
    builder's docstring states it, unstandardized by default
    (``standardize=False``)::

        p = z . b    loss = max(0, |y - p| - eps * sigma)    s = ||z||^2  (the intercept's 1 in)
        pa: tau = loss / s    pa1: tau = min(c, loss / s)    pa2: tau = loss / (s + 1 / (2c))
        b += min(w, 1) * tau * sign(y - p) * z

    per target, from zero. ``sigma`` is the target's own EW std as the row
    arrives: ``sigma ** 2`` the EW variance of ``y`` around its EW mean over
    the rows with the target and a weight above 0, whatever the fit, each
    ``y`` joining after its own row is judged, every weight aged by every
    row's decay, written here as its definition, two passes over the
    history; before it is above 0 the tube has no width (docs/PLAN.md task
    202). A null target or a zero weight moves nothing; the
    coefficients never decay, ``weight_sum`` does. ``pred_j`` is null while the
    target's own weight, the rows that carried it, decayed, is below
    ``min_weight`` (hard rule 8, docs/PLAN.md task 115 (d)); ``weight_sum`` is
    every row's. Returns ``pred``, ``weight_sum`` and ``coef`` (after the row).

    Under ``standardize``, ``z`` is the row standardized against the
    features' EW moments with the row admitted at unit weight
    (:func:`_row_map`), the moments learning each row at its weight. While
    Kish's count of the weights the moments have learned, undecayed, is
    below 22, ``b`` is in those coordinates, ``p = z . b``, and ``coef`` is it
    read out with the moments as they stand (:func:`_read_out`); on the row
    the count reaches 22, ``b`` is read out so once and held in the caller's
    units, and from the next row ``p = [1, x] . b`` and the step is mapped
    by the row's own ``m`` and ``s`` (:func:`_mapped`; docs/PLAN.md task
    206). ``decay_lam``, where given, is the decay per clock unit in place
    of ``half_life``'s. With ``coef_min``, ``coef_max`` or ``coef_sum`` the
    slopes are projected (:func:`_projected_standardized`,
    :func:`_projected_in_units`; Euclidean without a scaler) at the start,
    after each step and, while the scaler warms up, after every row that
    moves it."""
    n, k = X.shape
    m = Y.shape[1]
    kt = k + 1 if fit_intercept else k
    pred = np.full((n, m), np.nan)
    weight_sum = np.full(n, np.nan)
    coef = np.full((n, m, kt), np.nan)
    box = _box(k, coef_min, coef_max, coef_sum)
    b = np.zeros((m, kt))
    if box is not None:
        ones = np.ones(k)
        b = np.array([_projected_standardized(r, box, ones, fit_intercept) for r in b])
    w_sum = 0.0
    w_target = np.zeros(m)
    history: list[list[list[float]]] = [[] for _ in range(m)]
    rows: list[np.ndarray] = []
    om: list[float] = []
    kish_w, kish_w2, held = 0.0, 0.0, False
    for i, d in _accepted_steps(X, dclock, gap_cap):
        lam = _decay(half_life, decay_lam, d)
        xt = np.concatenate(([1.0], X[i])) if fit_intercept else X[i]
        z = xt
        if standardize:
            om = [o * lam for o in om]
            z, mv, sv = _row_map(rows, om, X[i], fit_intercept)
        weight_sum[i] = w_sum
        p = b @ (xt if (held or not standardize) else z)
        pred[i] = np.where(w_target >= min_weight, p, np.nan)
        s = z @ z
        stepped = np.zeros(m, dtype=bool)
        for j in range(m):
            sigma2 = _ew_variance(history[j])
            _age_and_learn(history[j], lam, Y[i, j], w[i])
            if np.isnan(Y[i, j]) or not w[i] > 0.0:
                continue
            r = Y[i, j] - p[j]
            tube = eps * np.sqrt(sigma2) if sigma2 > 0.0 else 0.0
            if s <= 0.0:
                continue
            loss = max(0.0, abs(r) - tube)
            if loss == 0.0:
                continue
            tau = {
                "pa": loss / s,
                "pa1": min(c, loss / s),
                "pa2": loss / (s + 1.0 / (2.0 * c)),
            }[mode]
            step = min(w[i], 1.0) * tau * np.sign(r) * z
            if held:
                b[j] += _mapped(step, mv, sv, fit_intercept)
                if box is not None:
                    b[j] = _projected_in_units(b[j], box, mv, sv, fit_intercept)
            else:
                b[j] += step
                stepped[j] = True
        if standardize:
            rows.append(X[i].copy())
            om.append(float(w[i]))
        if box is not None and not held:
            # In the scaler's coordinates the bounds move with the scales:
            # every target is projected when a row moved them.
            if standardize:
                _, s_now = _moments(rows, om, fit_intercept)
                for j in range(m):
                    if w[i] > 0.0 or stepped[j]:
                        b[j] = _projected_standardized(b[j], box, s_now, fit_intercept)
            else:
                for j in np.flatnonzero(stepped):
                    b[j] = _projected_standardized(b[j], box, np.ones(k), fit_intercept)
        if standardize and not held:
            kish_w += w[i]
            kish_w2 += w[i] ** 2
            m_now, s_now = _moments(rows, om, fit_intercept)
            if kish_w2 > 0.0 and kish_w**2 / kish_w2 >= WARMUP_ROWS:
                held = True
                b = np.array([_read_out(r, m_now, s_now, fit_intercept) for r in b])
        w_sum = lam * w_sum + w[i]
        w_target = lam * w_target + np.where(np.isnan(Y[i]), 0.0, w[i])
        if standardize and not held:
            m_now, s_now = _moments(rows, om, fit_intercept)
            coef[i] = np.array([_read_out(r, m_now, s_now, fit_intercept) for r in b])
        else:
            coef[i] = b
    return {"pred": pred, "weight_sum": weight_sum, "coef": coef}


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
    huber_delta: float = 1.345,
    quantile: float = 0.5,
    eps: float = 0.01,
    half_life: float = np.inf,
    fit_intercept: bool = True,
    min_weight: float = 0.0,
    gap_cap: float = np.inf,
    l2: float = 0.0,
    clip_gradient: float = np.inf,
    standardize: bool = False,
    decay_lam: float | None = None,
    coef_min: float | None = None,
    coef_max: float | None = None,
    coef_sum: float | None = None,
) -> dict[str, np.ndarray]:
    """Stochastic gradient descent as the ``sgd`` builder's docstring states
    it, unstandardized by default (``standardize=False``)::

        eta = z . b    p = link(eta)    d = dL / d eta
        squared: p - y                huber: clamp(p - y, +/- delta * s)
        quantile: 1{y < p} - tau      epsilon_insensitive: 0 within eps * s_y, else sign(p - y)
        poisson: exp(eta) - y         logistic: sigmoid(eta) - clamp(y, 0, 1)
        g_i = clamp(d * z_i * w + l2 * b_i, +/- clip_gradient)    b_i -= lr_i * g_i

    ``s`` is the EW std of the target's out-of-sample residuals as the row
    arrives: ``s ** 2`` the EW mean of ``(y - p) ** 2`` over the rows with the
    target, a weight above 0 and a prediction, each joining after its own
    step, its weight aged by every row's decay (docs/PLAN.md task 195).
    ``s_y`` is the target's own EW std as the row arrives, as ``pa_ref``
    keeps it (docs/PLAN.md task 202). Before ``s`` is above 0 the Huber loss
    cuts nothing, and before ``s_y`` is the tube has no width.

    the ridge on the slopes only (the intercept's ``g_0 = clamp(d * w)``),
    and the clip a cap on each coordinate of the gradient, not on its norm
    (review 2026-10-05, TC6). ``lr`` is ``learning_rate`` for ``"constant"``, ``learning_rate / (1 +
    weight_sum) ** power`` for ``"inv_scaling"`` with ``weight_sum`` the weight before
    the row, and ``learning_rate / (sqrt(G_i) + 1e-8)`` for ``"adagrad"``,
    ``G_i`` the sum of squared gradients with this row's in it. ``weight_sum``
    and ``G`` decay on the clock; the coefficients do not. Per target, from
    zero; a null target or a zero weight moves nothing. ``pred_j`` is ``p``,
    null while the target's own weight, the rows that carried it, decayed,
    is below ``min_weight`` (hard rule 8, docs/PLAN.md task 115 (d));
    ``weight_sum`` is every row's. Returns ``pred``, ``weight_sum`` and ``coef``
    (after the row), and ``clipped``, how many coordinates the clip bound.

    Under ``standardize``, ``z`` is the row standardized against the
    features' EW moments with the row admitted at unit weight
    (:func:`_row_map`), the moments learning each row at its weight. While
    Kish's count of the weights the moments have learned, undecayed, is
    below 22, ``b`` is in those coordinates, ``eta = z . b``, the ridge is on
    ``b`` there and ``coef`` is ``b`` read out with the moments as they stand
    (:func:`_read_out`); on the row the count reaches 22, ``b`` is read out
    so once and held in the caller's units, and from the next row ``eta =
    [1, x] . b``, the gradient is taken on ``z`` with the ridge on ``s_i *
    b_i``, and the step ``-lr_i * g_i`` is mapped by the row's own ``m`` and
    ``s`` (:func:`_mapped`; docs/PLAN.md task 206). ``decay_lam``, where
    given, is the decay per clock unit in place of ``half_life``'s. With
    ``coef_min``, ``coef_max`` or ``coef_sum`` the slopes are projected as
    :func:`pa_ref` projects them."""
    n, k = X.shape
    m = Y.shape[1]
    kt = k + 1 if fit_intercept else k
    pred = np.full((n, m), np.nan)
    weight_sum = np.full(n, np.nan)
    coef = np.full((n, m, kt), np.nan)
    box = _box(k, coef_min, coef_max, coef_sum)
    b = np.zeros((m, kt))
    if box is not None:
        ones = np.ones(k)
        b = np.array([_projected_standardized(r, box, ones, fit_intercept) for r in b])
    rows: list[np.ndarray] = []
    om: list[float] = []
    kish_w, kish_w2, held = 0.0, 0.0, False
    G = np.zeros((m, kt))
    penalised = np.ones(kt)
    if fit_intercept:
        penalised[0] = 0.0
    clipped = 0
    w_sum = 0.0
    w_target = np.zeros(m)
    sig2, wsig = np.zeros(m), np.zeros(m)
    history: list[list[list[float]]] = [[] for _ in range(m)]
    links = {
        "poisson": np.exp,
        "logistic": lambda e: 1.0 / (1.0 + np.exp(-e)),
    }
    for i, d in _accepted_steps(X, dclock, gap_cap):
        lam = _decay(half_life, decay_lam, d)
        G *= lam
        wsig *= lam
        xt = np.concatenate(([1.0], X[i])) if fit_intercept else X[i]
        z, scale = xt, np.ones(kt)
        if standardize:
            om = [o * lam for o in om]
            z, mv, scale = _row_map(rows, om, X[i], fit_intercept)
        weight_sum[i] = w_sum
        p = links.get(loss, lambda e: e)(b @ (xt if (held or not standardize) else z))
        pred[i] = np.where(w_target >= min_weight, p, np.nan)
        stepped = np.zeros(m, dtype=bool)
        for j in range(m):
            s_y2 = _ew_variance(history[j])
            _age_and_learn(history[j], lam, Y[i, j], w[i])
            if np.isnan(Y[i, j]) or not w[i] > 0.0:
                continue
            yj = min(max(Y[i, j], 0.0), 1.0) if loss == "logistic" else Y[i, j]
            e = p[j] - yj
            s = np.sqrt(sig2[j]) if sig2[j] > 0.0 else None
            cut = huber_delta * s if s is not None else np.inf
            tube = eps * np.sqrt(s_y2) if s_y2 > 0.0 else 0.0
            dl = {
                "squared": e,
                "poisson": e,
                "logistic": e,
                "huber": min(max(e, -cut), cut),
                "quantile": float(yj < p[j]) - quantile,
                "epsilon_insensitive": 0.0 if abs(e) <= tube else float(np.sign(e)),
            }[loss]
            if np.isfinite(pred[i, j]):
                sig2[j] = (wsig[j] * sig2[j] + w[i] * e * e) / (wsig[j] + w[i])
                wsig[j] += w[i]
            # The ridge on the slope in the row's coordinates: `b` itself
            # while it is held there, `s_i * b_i` once it is in the caller's
            # units.
            g = dl * z * w[i] + l2 * penalised * (scale * b[j] if held else b[j])
            clipped += int((np.abs(g) > clip_gradient).sum())
            g = np.clip(g, -clip_gradient, clip_gradient)
            if schedule == "constant":
                lr = learning_rate
            elif schedule == "inv_scaling":
                lr = learning_rate / (1.0 + w_sum) ** power
            else:
                G[j] += g * g
                lr = learning_rate / (np.sqrt(G[j]) + 1e-8)
            if held:
                b[j] += _mapped(-lr * g, mv, scale, fit_intercept)
                if box is not None:
                    b[j] = _projected_in_units(b[j], box, mv, scale, fit_intercept)
            else:
                b[j] -= lr * g
                stepped[j] = True
        if standardize:
            rows.append(X[i].copy())
            om.append(float(w[i]))
        if box is not None and not held:
            # In the scaler's coordinates the bounds move with the scales:
            # every target is projected when a row moved them.
            if standardize:
                _, s_now = _moments(rows, om, fit_intercept)
                for j in range(m):
                    if w[i] > 0.0 or stepped[j]:
                        b[j] = _projected_standardized(b[j], box, s_now, fit_intercept)
            else:
                for j in np.flatnonzero(stepped):
                    b[j] = _projected_standardized(b[j], box, np.ones(k), fit_intercept)
        if standardize and not held:
            kish_w += w[i]
            kish_w2 += w[i] ** 2
            m_now, s_now = _moments(rows, om, fit_intercept)
            if kish_w2 > 0.0 and kish_w**2 / kish_w2 >= WARMUP_ROWS:
                held = True
                b = np.array([_read_out(r, m_now, s_now, fit_intercept) for r in b])
        w_sum = lam * w_sum + w[i]
        w_target = lam * w_target + np.where(np.isnan(Y[i]), 0.0, w[i])
        if standardize and not held:
            m_now, s_now = _moments(rows, om, fit_intercept)
            coef[i] = np.array([_read_out(r, m_now, s_now, fit_intercept) for r in b])
        else:
            coef[i] = b
    return {"pred": pred, "weight_sum": weight_sum, "coef": coef, "clipped": clipped}


def holt_ref(
    Y: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    *,
    half_life: float,
    trend_half_life: float | None = None,
    min_weight: float = 0.0,
    gap_cap: float = np.inf,
) -> dict[str, np.ndarray]:
    """Holt's linear trend as the ``holt`` builder's docstring states it, per
    target, with ``s`` the clock since the target was last observed (this
    row's step included)::

        pred = l + b * s
        l'   = (lam_l * W * pred + w * y) / (lam_l * W + w)      W' = lam_l * W + w
        b'   = (lam_b * V * b + w * (l' - l) / s) / (lam_b * V + w)      V' = lam_b * V + w

    ``lam_l = 0.5 ** (s / half_life)`` and ``lam_b`` likewise at
    ``trend_half_life``, default four times the level's. The first
    observation sets the level; the second, at gain 1, the trend. A row at
    the last observation's clock (``s = 0``) is a second observation the
    level takes in, and the trend holds. A null target or a zero weight
    leaves ``l``, ``b``, ``W`` and ``V`` where they were and carries its
    clock to the next observation. ``pred`` is emitted from the second
    observation's row on, once the target's weight -- the rows it was
    present on, at their raw weights, decayed at the level half-life and seen
    before the row's own step -- reaches ``min_weight``; ``weight_sum`` is every
    row's weight so decayed. Each clock step is capped at ``gap_cap``.
    Returns ``pred``, ``weight_sum`` and ``coef`` (``[level, trend]`` after the
    row, NaN until the first observation)."""
    n, m = Y.shape
    th = 4.0 * half_life if trend_half_life is None else trend_half_life

    def lam(s: float, h: float) -> float:
        return 1.0 if np.isinf(h) else 0.5 ** (s / h)

    pred = np.full((n, m), np.nan)
    weight_sum = np.full(n, np.nan)
    coef = np.full((n, m, 2), np.nan)
    level = np.full(m, np.nan)
    trend = np.zeros(m)
    W = np.zeros(m)
    V = np.zeros(m)
    since = np.zeros(m)
    seen = np.zeros(m, dtype=int)
    w_sum, w_own = 0.0, np.zeros(m)
    for i in range(n):
        d = 0.0 if i == 0 else min(max(t[i] - t[i - 1], 0.0), gap_cap)
        weight_sum[i] = w_sum
        for j in range(m):
            since[j] += d
            s = since[j]
            if seen[j] >= 1 and w_own[j] >= min_weight and w_own[j] > 0.0:
                pred[i, j] = level[j] + trend[j] * s
            if np.isnan(Y[i, j]) or not w[i] > 0.0:
                continue
            if seen[j] == 0:
                level[j], W[j] = Y[i, j], w[i]
            else:
                p = level[j] + trend[j] * s
                ll = lam(s, half_life)
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
        step = lam(d, half_life)
        w_sum = step * w_sum + w[i]
        w_own = step * w_own + w[i] * ~np.isnan(Y[i])
        for j in range(m):
            if seen[j]:
                coef[i, j] = (level[j], trend[j])
    return {"pred": pred, "weight_sum": weight_sum, "coef": coef}
