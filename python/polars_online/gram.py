"""The accumulators, read back (docs/ENHANCEMENTS.md E46).

:meth:`~polars_online.ModelBank.gram` hands back the matrices the models
solve against, from one pass over data that is never materialized. This
module is what to do with them afterwards: pool shards, take a subset, read a
correlation, solve a ridge, walk a lasso path, put standard errors on
coefficients, and diagnose collinearity.

Every function takes the mapping ``gram()`` produces -- ``columns``,
``means``, ``comoments``, ``cross_moments``, ``target_weights``,
``target_means``, ``target_vars``, ``n_eff``, ``n_kish``, ``target_n_kish``
-- and :func:`merge` and :func:`subset` return one of the same shape.

The arithmetic is the models' own, so :func:`solve` on a spec's Gram
reproduces that spec's coefficients and :func:`lasso_path` reproduces the
``lasso`` model's path. What it is not is the *same* arithmetic to the last
bit: the models factorize with ``faer``'s Cholesky and numpy with LAPACK's
LU, which round differently in the last place or two. The tests hold the two
to a relative tolerance, not to equality, and say so.

Requires numpy, which is an optional extra of this package
(``pip install polars-online[numpy]``) -- not a dependency, as it is not one
of polars' either. Nothing here needs scipy or scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "coef_stats",
    "condition",
    "correlation",
    "lasso_path",
    "merge",
    "solve",
    "subset",
    "vif",
]

#: The name :meth:`~polars_online.ModelBank.gram` gives the constant column a
#: spec's ``add_intercept`` puts in front of the features, matching the
#: ``term`` column of :func:`polars_online.spec.coef_index`.
INTERCEPT = "intercept"


def _np() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as e:  # pragma: no cover - exercised by a stub
        msg = (
            "polars_online.gram works in numpy arrays, and numpy is not installed. "
            "Install it with `pip install numpy` or `pip install polars-online[numpy]`."
        )
        raise ModuleNotFoundError(msg) from e
    return np


def _columns(g: dict[str, Any]) -> list[str]:
    cols = g.get("columns")
    if cols is None:
        msg = (
            "this mapping has no 'columns': polars_online.gram works on what "
            "ModelBank.gram() returns, which names its columns"
        )
        raise KeyError(msg)
    return list(cols)


def _col_index(g: dict[str, Any], cols: Sequence[str | int]) -> list[int]:
    names = _columns(g)
    out = []
    for c in cols:
        if isinstance(c, int):
            if not -len(names) <= c < len(names):
                msg = f"column {c} out of range for a Gram of {len(names)} columns"
                raise IndexError(msg)
            out.append(c % len(names))
        elif c in names:
            out.append(names.index(c))
        else:
            msg = f"no column {c!r} in this Gram; it has {names}"
            raise KeyError(msg)
    return out


def _target_index(g: dict[str, Any], target: str | int) -> int:
    names = list(g.get("targets") or [])
    if isinstance(target, int):
        n = len(g["cross_moments"])
        if not -n <= target < n:
            msg = f"target {target} out of range for a Gram with {n} targets"
            raise IndexError(msg)
        return target % n
    if target not in names:
        msg = f"no target {target!r} in this Gram; it has {names}"
        raise KeyError(msg)
    return names.index(target)


def _feature_slots(
    g: dict[str, Any], features: Sequence[str | int] | None
) -> tuple[list[int], int]:
    """The column positions to regress on, and the intercept's position or -1.

    The intercept is never one of the features: it is a constant column with
    zero variance, and treating it as a regressor is how a solve ends up
    singular.
    """
    names = _columns(g)
    icept = names.index(INTERCEPT) if INTERCEPT in names else -1
    if features is None:
        slots = [i for i in range(len(names)) if i != icept]
    else:
        slots = _col_index(g, features)
        if icept in slots:
            msg = (
                f"{INTERCEPT!r} is a constant column, not a feature; it is "
                "handled by the solve, so leave it out of `features`"
            )
            raise ValueError(msg)
    return slots, icept


def merge(grams: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool the Grams of **disjoint row sets** into the Gram of their union.

    Chan, Golub and LeVeque's update: the pooled co-moments are the weighted
    average of the parts' plus the spread *between* their means, and every
    quantity is a sum of parts rather than a difference of cumulative sums --
    so pooling a thousand shards loses no more precision than pooling two.
    With weights ``W_a``, ``W_b`` and mean gap ``d = m_b - m_a``::

        W = W_a + W_b
        m = m_a + (W_b / W) * d
        C = (W_a * C_a + W_b * C_b) / W + (W_a * W_b / W**2) * outer(d, d)
        Q = Q_a + Q_b

    Use it to pool accumulators that share a weighting: one per shard of a
    pass, one per group being combined, one per worker. **Not** two halves of
    a decayed stream in time order -- each part's weights are relative to its
    own last row, so the earlier part is over-weighted by exactly the decay
    between them. Either run the parts under an infinite halflife, or scale
    the earlier part's ``n_eff`` by ``lam**dt`` and its ``sum(w**2)`` by
    ``lam**(2*dt)`` before merging (the means and co-moments are unaffected,
    being weighted means already).

    Every part must have the same ``columns`` and ``targets``; a part with no
    ``n_kish`` or no target moments (a state saved by 0.2.0 or earlier) makes
    the merge report ``None`` for those, since the sums behind them are not
    there to add. ``group`` and ``instance`` come back as ``None``: a pooled
    accumulator is no longer one group's or one instance's.

    Merging one Gram returns it unchanged; merging none is a ``ValueError``.
    """
    np = _np()
    parts = list(grams)
    if not parts:
        msg = "merge() needs at least one Gram"
        raise ValueError(msg)
    cols = _columns(parts[0])
    targets = list(parts[0].get("targets") or [])
    for p in parts[1:]:
        if _columns(p) != cols:
            msg = f"merge() needs the same columns in every part: {cols} vs {_columns(p)}"
            raise ValueError(msg)
        if list(p.get("targets") or []) != targets:
            msg = "merge() needs the same targets in every part"
            raise ValueError(msg)

    w = float(parts[0]["n_eff"])
    mean = np.asarray(parts[0]["means"], dtype=float).copy()
    como = np.asarray(parts[0]["comoments"], dtype=float).copy()
    q = _q_of(parts[0])
    tw = np.asarray(parts[0]["target_weights"], dtype=float).copy()
    cross = np.asarray(parts[0]["cross_moments"], dtype=float).copy()
    tmean = _opt(np, parts[0]["target_means"])
    tvar = _opt(np, parts[0]["target_vars"])
    tq = _target_q(np, parts[0])

    for p in parts[1:]:
        wb = float(p["n_eff"])
        total = w + wb
        if total > 0.0:
            mb = np.asarray(p["means"], dtype=float)
            d = mb - mean
            cb = np.asarray(p["comoments"], dtype=float)
            como = (w * como + wb * cb) / total + (w * wb / total**2) * np.outer(d, d)
            mean = mean + (wb / total) * d
        w = total
        q = None if q is None else _add_opt(q, _q_of(p))

        twb = np.asarray(p["target_weights"], dtype=float)
        crossb = np.asarray(p["cross_moments"], dtype=float)
        tmb, tvb = _opt(np, p["target_means"]), _opt(np, p["target_vars"])
        ttotal = tw + twb
        live = ttotal > 0.0
        if tmean is not None and tmb is not None and tvar is not None and tvb is not None:
            d = np.where(live, tmb - tmean, 0.0)
            a = np.divide(tw, ttotal, out=np.zeros_like(ttotal), where=live)
            b = np.divide(twb, ttotal, out=np.zeros_like(ttotal), where=live)
            tvar = a * tvar + b * tvb + a * b * d * d
            tmean = tmean + b * d
        else:
            tmean = tvar = None
        tq = None if tq is None else _add_opt(tq, _target_q(np, p))
        if cross.size or crossb.size:
            scale = np.divide(1.0, ttotal, out=np.zeros_like(ttotal), where=live)
            cross = (tw[:, None] * cross + twb[:, None] * crossb) * scale[:, None]
        tw = ttotal

    return {
        "group": None,
        "instance": None,
        "columns": cols,
        "targets": targets,
        "n_eff": w,
        "n_kish": None if q is None or q <= 0.0 else w * w / q,
        "means": mean,
        "comoments": como,
        "cross_moments": cross,
        "target_weights": tw,
        "target_means": tmean,
        "target_vars": tvar,
        "target_n_kish": None
        if tq is None
        else np.divide(tw * tw, tq, out=np.full_like(tq, np.nan), where=tq > 0.0),
    }


def _opt(np: Any, v: Any) -> Any:
    return None if v is None else np.asarray(v, dtype=float).copy()


def _add_opt(a: Any, b: Any) -> Any:
    return None if b is None else a + b


def _q_of(g: dict[str, Any]) -> float | None:
    """`sum(w**2)` behind the feature moments, back out of `n_kish`."""
    nk = g.get("n_kish")
    if nk is None or not nk > 0.0:
        return None
    w = float(g["n_eff"])
    return w * w / float(nk)


def _target_q(np: Any, g: dict[str, Any]) -> Any:
    nk = g.get("target_n_kish")
    if nk is None:
        return None
    nk = np.asarray(nk, dtype=float)
    tw = np.asarray(g["target_weights"], dtype=float)
    return np.divide(tw * tw, nk, out=np.zeros_like(tw), where=np.isfinite(nk) & (nk > 0.0))


def subset(g: dict[str, Any], cols: Sequence[str | int]) -> dict[str, Any]:
    """The Gram of a subset of the columns, in the order given.

    Exact, not approximate: a marginal set of moments is a sub-block of the
    joint ones, so this is a selection rather than a recomputation, and a
    regression on the subset is the regression the full accumulator implies.
    That is the point -- forward stepwise, an information criterion over
    feature sets, or an ``r``-column fit read off a ``k``-column stream all
    fall out of one pass.

    Names or positions, and the intercept may be selected like any other
    column. Targets are untouched: they index a different axis.
    """
    np = _np()
    idx = _col_index(g, cols)
    names = _columns(g)
    como = np.asarray(g["comoments"], dtype=float)
    cross = np.asarray(g["cross_moments"], dtype=float)
    return {
        **g,
        "columns": [names[i] for i in idx],
        "means": np.asarray(g["means"], dtype=float)[idx],
        "comoments": como[np.ix_(idx, idx)],
        "cross_moments": cross[:, idx] if cross.size else cross,
    }


def correlation(g: dict[str, Any]) -> Any:
    """The correlation matrix of the columns, from the centred co-moments.

    ``nan`` in the row and column of a constant one (the intercept included:
    a constant has no correlation with anything, and reporting 0 there would
    read as "independent"). The diagonal is 1 where the variance is positive.
    """
    np = _np()
    c = np.asarray(g["comoments"], dtype=float)
    s = np.sqrt(np.clip(np.diag(c), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = c / np.outer(s, s)
    r[~np.isfinite(r)] = np.nan
    dead = s <= 0.0
    r[dead, :] = np.nan
    r[:, dead] = np.nan
    return r


def solve(
    g: dict[str, Any],
    *,
    ridge: float | Sequence[float] = 0.0,
    target: str | int = 0,
    features: Sequence[str | int] | None = None,
    standardize: bool = False,
) -> Any:
    """Ridge coefficients from the Gram, in the features' original units.

    The model's own algebra (``EwRidge::solve``), so the result is the fit
    that spec would report on the same accumulator:

    - ``standardize=False`` adds ``ridge`` to the diagonal of the *raw*
      second-moment matrix, leaving the intercept unpenalized;
    - ``standardize=True`` centres, scales to correlation form, adds ``ridge``
      there, then unscales and recovers the intercept from the means -- so
      ``ridge`` means the same thing whatever the features' units. A column
      with zero variance is dropped with a coefficient of 0 rather than
      making the system singular.

    Pass the ``standardize`` the spec used, or the numbers will not match its
    ``coef()``. With an intercept in ``columns`` the returned vector starts
    with it, in :func:`polars_online.spec.coef_index` order.

    ``ridge`` may be a sequence, and then the return is one row per value.
    A grid rides a single eigendecomposition wherever the penalty is uniform
    in the basis being solved -- always with ``standardize=True``, and
    without an intercept otherwise: with ``V d V'`` in hand every ridge is
    ``V diag(1/(d + r)) V' b``, which is what makes a grid of fifty cheap.
    An unstandardized fit *with* an intercept leaves that one column
    unpenalized, so its penalty is not a multiple of the identity and each
    value costs a factorization. That is the model's arithmetic, and
    reproducing it is worth more here than the shortcut.

    ``target`` picks the target by name or position; ``features`` narrows the
    regressors (equivalent to :func:`subset` first, and refusing the
    intercept, which the solve handles itself).
    """
    np = _np()
    t = _target_index(g, target)
    slots, icept = _feature_slots(g, features)
    ridges = np.atleast_1d(np.asarray(ridge, dtype=float))
    scalar = np.ndim(ridge) == 0
    k = len(_columns(g))

    means = np.asarray(g["means"], dtype=float)
    cross = np.asarray(g["cross_moments"], dtype=float)[t]
    como = np.asarray(g["comoments"], dtype=float)
    out = np.zeros((len(ridges), k))

    if not standardize:
        # Raw second moments, as the model pairs them with the uncentred
        # cross-moments: raw = comoments + outer(means, means).
        zidx = ([icept] if icept >= 0 else []) + slots
        raw = como[np.ix_(zidx, zidx)] + np.outer(means[zidx], means[zidx])
        b = cross[zidx]
        pen = np.ones(len(zidx))
        if icept >= 0:
            pen[0] = 0.0  # the intercept is not shrunk
        d, v = np.linalg.eigh(raw)
        vb = v.T @ b
        for i, r in enumerate(ridges):
            # The penalty is diagonal in the original basis, not the
            # eigenbasis, so only a uniform one can ride the decomposition.
            if icept >= 0 and r != 0.0:
                sol = np.linalg.solve(raw + r * np.diag(pen), b)
            else:
                sol = v @ (vb / (d + r))
            out[i, zidx] = sol
        return out[0] if scalar else out

    # Standardized: centre, scale to correlation form, solve, unscale, then
    # recover the intercept from the means. A constant column is dropped.
    c = como[np.ix_(slots, slots)]
    s = np.sqrt(np.clip(np.diag(c), 0.0, None))
    keep = [i for i in range(len(slots)) if s[i] > 0.0]
    ybar = cross[icept] if icept >= 0 else 0.0
    if keep:
        kk = np.ix_(keep, keep)
        a = c[kk] / np.outer(s[keep], s[keep])
        b = (cross[[slots[i] for i in keep]] - means[[slots[i] for i in keep]] * ybar) / s[keep]
        d, v = np.linalg.eigh(a)
        vb = v.T @ b
        for i, r in enumerate(ridges):
            sol = v @ (vb / (d + r))
            out[i, [slots[j] for j in keep]] = sol / s[keep]
    if icept >= 0:
        out[:, icept] = ybar - out[:, slots] @ means[slots]
    return out[0] if scalar else out


def lasso_path(
    g: dict[str, Any],
    lambdas: Sequence[float],
    *,
    l1_ratio: float = 1.0,
    penalty_weights: Sequence[float] | None = None,
    target: str | int = 0,
    features: Sequence[str | int] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> Any:
    """The elastic-net path from the Gram, one row of coefficients per lambda.

    The ``lasso`` model's coordinate descent (``Lasso::solve``), run offline:
    on the standardized (correlation-form) matrix, warm-started down the path
    in the order given, with

    ``b_i = soft(rho_i, l * l1_ratio * pw_i) / (C_ii + l * (1 - l1_ratio) * pw_i)``

    where ``rho_i`` is the standardized cross-correlation less the other
    columns' contributions and ``soft(v, t) = sign(v) * max(|v| - t, 0)``.
    Coefficients come back in original units with the intercept recovered from
    the means, so a row is directly comparable to ``bank.coef()``.

    ``penalty_weights`` scales the penalty per feature (in ``features``
    order, or ``columns`` order without the intercept): 0 leaves a column
    unpenalized, and a column the stream found constant is dropped whatever
    is asked for. The online model has no such parameter -- it is the one
    thing here that the models do not also do, and it is cheap offline
    because the path is re-walked rather than carried.

    Give ``lambdas`` from large to small, as a path is meant to be walked:
    the warm start makes that both faster and better conditioned. ``max_iter``
    and ``tol`` are the model's ``max_cd_iters`` and ``cd_tol``.
    """
    np = _np()
    t = _target_index(g, target)
    slots, icept = _feature_slots(g, features)
    k = len(_columns(g))
    means = np.asarray(g["means"], dtype=float)
    cross = np.asarray(g["cross_moments"], dtype=float)[t]
    como = np.asarray(g["comoments"], dtype=float)

    c = como[np.ix_(slots, slots)]
    s = np.sqrt(np.clip(np.diag(c), 0.0, None))
    live = s > 0.0
    scale = np.where(live, s, 1.0)
    # The model writes 1 on the diagonal of a dead column so the descent
    # divides by something; its coefficient is pinned at 0 regardless.
    corr = c / np.outer(scale, scale)
    corr[~live, :] = 0.0
    corr[:, ~live] = 0.0
    corr[~live, ~live] = 1.0
    ybar = cross[icept] if icept >= 0 else 0.0
    d = np.where(live, (cross[slots] - means[slots] * ybar) / scale, 0.0)

    pw = (
        np.ones(len(slots)) if penalty_weights is None else np.asarray(penalty_weights, dtype=float)
    )
    if pw.shape != (len(slots),):
        msg = f"penalty_weights must have one entry per feature ({len(slots)}), got {pw.shape}"
        raise ValueError(msg)

    out = np.zeros((len(lambdas), k))
    b = np.zeros(len(slots))
    for li, lam in enumerate(lambdas):
        l1, l2 = lam * l1_ratio * pw, lam * (1.0 - l1_ratio) * pw
        for _ in range(max_iter):
            delta = 0.0
            for i in range(len(slots)):
                if not live[i]:
                    b[i] = 0.0
                    continue
                rho = d[i] - (corr[i] @ b - corr[i, i] * b[i])
                new = np.sign(rho) * max(abs(rho) - l1[i], 0.0) / (corr[i, i] + l2[i])
                delta = max(delta, abs(new - b[i]))
                b[i] = new
            if delta < tol:
                break
        out[li, slots] = np.where(live, b / scale, 0.0)
        if icept >= 0:
            out[li, icept] = ybar - out[li, slots] @ means[slots]
    return out


def coef_stats(
    g: dict[str, Any],
    coef: Sequence[float],
    *,
    target: str | int = 0,
    features: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    """Residual variance, standard errors and t-statistics for a fit.

    This is what the target moments were added for (E45): with ``Var[y]`` in
    the Gram, a saved state answers "how good is this fit, and which
    coefficients are real" without the rows::

        resid_var = Var[y] - 2 b' Cov[X, y] + b' C b
        sigma2    = resid_var * n / (n - k)          # n = target_n_kish
        se        = sqrt(diag(inv(C)) * sigma2 / n)
        t         = b / se

    Returns ``resid_var``, ``sigma2``, ``r2``, ``n`` (the Kish size the
    correction and the errors use), ``se`` and ``t`` -- the last two arrays
    over the same slots as ``coef``, with the intercept's entry ``nan``
    (its standard error depends on the design's centring, which the Gram has
    already absorbed).

    ``n`` is Kish's effective sample size, not ``n_eff``: a weighted stream's
    weight sum is not a count, and dividing by it would report standard
    errors too small by the factor the weights are unequal by. The rows
    behind an exponentially weighted fit are also neither independent nor
    identically distributed, so read a ``t`` here as a scale for comparing
    coefficients, not as a p-value.

    ``ValueError`` if the Gram has no target moments -- a state saved by
    0.2.0 or earlier cannot answer this.
    """
    np = _np()
    t = _target_index(g, target)
    slots, icept = _feature_slots(g, features)
    if g.get("target_vars") is None or g.get("target_n_kish") is None:
        msg = (
            "this Gram has no target moments, so it has no Var[y] to take a "
            "residual variance from; a state saved by 0.2.0 or earlier reports "
            "None for target_vars and target_n_kish (ENHANCEMENTS E45)"
        )
        raise ValueError(msg)

    beta = np.asarray(coef, dtype=float)
    k = len(_columns(g))
    if beta.shape != (k,):
        msg = f"coef must have one entry per Gram column ({k}), got {beta.shape}"
        raise ValueError(msg)
    b = beta[slots]
    means = np.asarray(g["means"], dtype=float)
    cross = np.asarray(g["cross_moments"], dtype=float)[t]
    como = np.asarray(g["comoments"], dtype=float)
    c = como[np.ix_(slots, slots)]
    var_y = float(np.asarray(g["target_vars"], dtype=float)[t])
    n = float(np.asarray(g["target_n_kish"], dtype=float)[t])
    ybar = cross[icept] if icept >= 0 else float(np.asarray(g["target_means"], dtype=float)[t])
    # Centred cross-covariance: E[z y] - E[z] E[y], the pair `comoments` is in.
    cov_xy = cross[slots] - means[slots] * ybar

    resid_var = var_y - 2.0 * b @ cov_xy + b @ c @ b
    resid_var = max(resid_var, 0.0)
    dof = n - (len(slots) + (1 if icept >= 0 else 0))
    sigma2 = resid_var * n / dof if dof > 0.0 else np.nan
    se = np.full(k, np.nan)
    tstat = np.full(k, np.nan)
    if np.isfinite(sigma2) and n > 0.0:
        try:
            inv = np.linalg.inv(c)
        except np.linalg.LinAlgError:
            inv = np.full_like(c, np.nan)
        se[slots] = np.sqrt(np.clip(np.diag(inv), 0.0, None) * sigma2 / n)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat[slots] = np.where(se[slots] > 0.0, beta[slots] / se[slots], np.nan)
    return {
        "resid_var": resid_var,
        "sigma2": sigma2,
        "r2": 1.0 - resid_var / var_y if var_y > 0.0 else float("nan"),
        "n": n,
        "se": se,
        "t": tstat,
    }


def vif(g: dict[str, Any], *, features: Sequence[str | int] | None = None) -> Any:
    """Variance inflation factors: ``1 / (1 - R2_j)`` for each column on the
    rest, straight off the diagonal of the inverse correlation matrix.

    The intercept is not a regressor and is left out by default (its VIF is
    undefined -- a constant is perfectly explained by any other constant).
    A column the stream found constant reports ``inf``.

    Above about 10 the coefficient of that column is mostly noise; the fix is
    a ridge, a subset, or a feature set the spec already knows how to fit
    beside the full one.
    """
    np = _np()
    slots, _ = _feature_slots(g, features)
    r = correlation(g)[np.ix_(slots, slots)]
    if not np.all(np.isfinite(r)):
        out = np.full(len(slots), np.inf)
        ok = [i for i in range(len(slots)) if np.isfinite(r[i]).all()]
        if ok:
            out[ok] = np.diag(np.linalg.pinv(r[np.ix_(ok, ok)]))
        return out
    return np.diag(np.linalg.pinv(r))


def condition(g: dict[str, Any], *, features: Sequence[str | int] | None = None) -> dict[str, Any]:
    """Belsley's collinearity diagnostics for the accumulated design.

    Returns ``singular_values`` (of the column-scaled design, largest first),
    ``condition_indexes`` (``s_max / s_j``), ``kappa`` (the largest of them)
    and ``proportions`` -- the variance-decomposition proportions, one row per
    component and one column per feature, each column summing to 1.

    A component with a large condition index *and* a large share of two or
    more columns' variance is a near-dependency between exactly those
    columns, which is what makes this worth more than a single ``kappa``: it
    says which columns are the problem, where a VIF only says that one is.
    Belsley's rule of thumb is an index above 30 with two proportions above
    0.5.

    The design is scaled to unit column length first (Belsley's
    prescription), but *not* centred: the intercept is part of the
    collinearity when a column is nearly constant, and centring hides that.
    ``singular_values`` are of that scaled raw matrix, so they are the square
    roots of the eigenvalues of the scaled second-moment matrix.
    """
    np = _np()
    names = _columns(g)
    slots = list(range(len(names))) if features is None else _col_index(g, features)
    means = np.asarray(g["means"], dtype=float)
    como = np.asarray(g["comoments"], dtype=float)
    raw = (como + np.outer(means, means))[np.ix_(slots, slots)]
    scale = np.sqrt(np.clip(np.diag(raw), 0.0, None))
    scale = np.where(scale > 0.0, scale, 1.0)
    a = raw / np.outer(scale, scale)
    d, v = np.linalg.eigh(a)
    order = np.argsort(d)[::-1]
    d, v = np.clip(d[order], 0.0, None), v[:, order]
    sv = np.sqrt(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = np.where(sv > 0.0, sv[0] / sv, np.inf)
        # phi_{ji} = v_{ij}^2 / d_j, proportions normalized down each column.
        phi = (v.T**2) / np.where(d[:, None] > 0.0, d[:, None], np.nan)
        props = phi / np.nansum(phi, axis=0)
    return {
        "columns": [names[i] for i in slots],
        "singular_values": sv,
        "condition_indexes": idx,
        "kappa": float(idx[-1]) if len(idx) else float("nan"),
        "proportions": props,
    }
