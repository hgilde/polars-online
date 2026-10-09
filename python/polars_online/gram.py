"""The running sums a bank exports, read back: pool them, subset them, solve
them, and put standard errors on a fit.

:meth:`~polars_online.ModelBank.gram` hands back the matrices the models solve
against, from one pass over data that is never materialized. This module is
what to do with them afterwards: pool shards (:func:`merge`), take a subset of
the columns (:func:`subset`), read a correlation matrix (:func:`correlation`),
solve a ridge (:func:`solve`), walk a lasso path over a grid of penalties
(:func:`lasso_path`) or from its start knot by knot (:func:`lars_path`, and
:func:`lars_paths` for many Grams and targets at once), put standard errors
on coefficients (:func:`coef_stats`), merge and fit many subsets of Grams in
one call (:func:`solve_subsets`), and diagnose collinearity (:func:`vif`,
:func:`condition`).

Every function takes the mapping ``gram()`` produces (``columns``,
``targets``, ``means``, ``comoments``, ``cross_moments``, ``means_by_target``,
``cross_centred``, ``target_weights``, ``target_means``, ``target_vars``,
``weight_sum``, ``n_kish``, ``target_n_kish``), and :func:`merge`, :func:`subset`
and :func:`from_row` return one of the same shape, so a closed group's row
(:meth:`~polars_online.ModelBank.closed_groups`) is read the same way. The
compact forms ``gram()`` hands back under ``dtype="float32"`` or
``layout="packed"`` are taken too, and read in float64.

The arithmetic is the models' own, so :func:`solve` on a spec's Gram
reproduces that spec's coefficients and :func:`lasso_path` reproduces the
``lasso`` model's path. It is not the same arithmetic to the last bit: the
models factorize with ``faer``'s Cholesky, and here numpy runs LAPACK's
symmetric eigendecomposition (``numpy.linalg.eigh``) in :func:`solve` and
:func:`vif`, and an inverse (``numpy.linalg.inv``) in :func:`coef_stats`,
which round differently in the last place or two. The tests hold the two to
a relative tolerance, not to equality. :func:`lasso_path` and
:func:`lars_path` run in Rust, and :func:`lars_paths` and
:func:`solve_subsets` on the bank's thread pool.

Requires numpy, which is an optional extra of this package (``pip install
polars-online[numpy]``), not a dependency, as it is not one of polars' either.
Nothing here needs scipy or scikit-learn.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from polars_online import _polars_online as _native
from polars_online._renamed import renamed_keywords

__all__ = [
    "INTERCEPT",
    "coef_stats",
    "condition",
    "correlation",
    "from_row",
    "lars_path",
    "lars_paths",
    "lasso_path",
    "merge",
    "solve",
    "solve_subsets",
    "subset",
    "vif",
]

#: The name :meth:`~polars_online.ModelBank.gram` gives the constant column a
#: spec's ``fit_intercept`` puts in front of the features, matching the
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


def _means_of(np: Any, g: dict[str, Any], t: int) -> Any:
    """The column means target ``t``'s cross-moments are centred at and its
    intercept is recovered from: its own, over the rows it was present on
    (``means_by_target``, docs/PLAN.md task 81), or the Gram's ``means`` in a
    mapping without them."""
    by_target = g.get("means_by_target")
    if by_target is None or len(by_target) == 0:
        return np.asarray(g["means"], dtype=float)
    return np.asarray(by_target, dtype=float)[t]


def _cross_centred(np: Any, g: dict[str, Any], t: int, m: Any, ybar: float) -> Any:
    """Target ``t``'s cross-moments centred at its column means ``m`` and its
    mean ``ybar``, as the model holds them: ``cross_centred`` (review
    2026-09-12, N4), or -- in a mapping without it -- ``cross_moments[t] - m *
    ybar``, a difference of two numbers the size of ``L**2`` at a level ``L``,
    which keeps ``L**2 * eps`` of the answer."""
    cc = g.get("cross_centred")
    if cc is not None and len(cc):
        return np.asarray(cc, dtype=float)[t]
    return np.asarray(g["cross_moments"], dtype=float)[t] - m * ybar


def _ybar_of(np: Any, g: dict[str, Any], icept: int) -> Any:
    """Each target's mean, which its centred cross-moments are centred at:
    the raw cross-moment at the intercept, exactly (``0 + 1 * ybar``), or
    ``target_means``; ``None`` when the mapping has neither."""
    cross = np.asarray(g["cross_moments"], dtype=float)
    if icept >= 0 and cross.size:
        return cross[:, icept].copy()
    tm = g.get("target_means")
    return None if tm is None else np.asarray(tm, dtype=float).copy()


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


def _is_packed(np: Any, g: dict[str, Any]) -> bool:
    """Whether a Gram's ``comoments`` is the packed upper triangle
    (``ModelBank.gram(layout="packed")``), one axis, rather than the whole
    matrix."""
    return bool(np.ndim(g["comoments"]) == 1)


def _unpack(np: Any, v: Any, k: int) -> Any:
    """The symmetric ``k x k`` float64 matrix whose upper triangle, row by row,
    is ``v``: its upper triangle and diagonal bit for bit, its lower the mirror."""
    m = np.zeros((k, k))
    m[np.triu_indices(k)] = np.asarray(v, dtype=float)
    return m + np.triu(m, 1).T


def _comoments(np: Any, g: dict[str, Any]) -> Any:
    """``g``'s co-moments as the whole ``k x k`` matrix in float64, whatever
    form ``ModelBank.gram()`` handed them over in (``dtype``, ``layout``)."""
    if _is_packed(np, g):
        return _unpack(np, g["comoments"], len(_columns(g)))
    return np.asarray(g["comoments"], dtype=float)


def merge(grams: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool the Grams of disjoint row sets into the Gram of their union.

    Chan, Golub and LeVeque's update: the pooled co-moments are the weighted
    average of the parts' plus the spread between their means, and every quantity
    is a sum of parts rather than a difference of cumulative sums, so pooling a
    thousand shards loses no more precision than pooling two. With weights
    ``W_a``, ``W_b`` and mean gap ``d = m_b - m_a``:

    .. code-block:: text

        W = W_a + W_b
        m = m_a + (W_b / W) * d
        C = (W_a * C_a + W_b * C_b) / W + (W_a * W_b / W**2) * outer(d, d)
        Q = Q_a + Q_b

    Use it to pool accumulators that share a weighting: one per shard of a pass,
    one per group being combined, one per worker.

    To merge two halves of a decayed stream in time order, decay the earlier
    half to the later half's last row before merging. Each half's weights are
    relative to its own last row, so the earlier half is over-weighted by
    exactly the decay between them. Multiply the earlier half's ``weight_sum``
    and ``target_weights`` by ``0.5 ** (dt / half_life)``, with ``dt`` the clock
    from its last row to the later half's last row, each step capped at
    ``gap_cap`` as the bank caps it. Leave every other field as it is: the
    means and co-moments are weighted means, and ``n_kish`` and
    ``target_n_kish`` are scale-free, so the sums of squared weights the merge
    recovers from them decay with the weights. Halves run under
    ``half_life=float("inf")`` merge as they are.

    .. code-block:: python

        halves = po.spec.ewridge("h", targets=["y"], features=["x0", "x1"],
                                 clock="t", gap_cap=10.0, half_life=100.0)

        def gram_of(rows):                               # a fresh bank's Gram after these rows
            fitted = po.ModelBank([halves])
            fitted.fit_predict(rows)
            return fitted.gram("h")[0]

        early, late = gram_of(df.head(200)), gram_of(df.tail(200))
        dt = df["t"][399] - df["t"][199]                 # early half's last row to late half's
        decay = 0.5 ** (dt / 100.0)
        early = dict(early, weight_sum=early["weight_sum"] * decay,
                     target_weights=early["target_weights"] * decay)
        pooled = po.gram.merge([early, late])            # the Gram of one bank over all 400 rows

    Each target's ``means_by_target`` pools over its own rows, by its
    ``target_weights``, and its ``cross_centred`` as the co-moments do, with the
    gaps between the parts' column means and target means. A part without them
    makes the merge report ``None`` there, and :func:`solve` then forms them from
    ``cross_moments``. Under ``target_gaps = "own_rows"`` a spec may have a Gram
    per set of targets: merge the entries of one Gram across the shards, the ones
    with the same ``targets``.

    Returns a mapping of the same shape. ``lags`` and ``lag_comoments`` come back
    ``None``: a lagged cross-moment pairs a row with the row ``l`` back within its
    own part, and the pairings across a part boundary are what no part holds.
    ``group`` and ``instance`` come back ``None``, since a pooled accumulator
    belongs to no one group or instance. A part with no ``n_kish`` or no target
    moments (a state saved by 0.2.0 or earlier) makes the merge report ``None``
    for those, since the sums behind them are not there to add.

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0, group="stock_id"
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        parts = bank.gram("ridge")                      # one Gram per group
        pooled = po.gram.merge(parts)                   # the Gram of every group's rows together

    Parts in a compact form (``ModelBank.gram(dtype="float32")`` or
    ``layout="packed"``) merge in float64. When every part is packed the merge
    is packed too, entry for entry; otherwise it is the whole matrix.

    Every part must have the same ``columns`` and ``targets`` (``ValueError``).
    Merging one Gram returns it unchanged; merging none is a ``ValueError``.
    """
    np = _np()
    parts = list(grams)
    if not parts:
        msg = "merge() needs at least one Gram"
        raise ValueError(msg)
    if len(parts) == 1:
        # Nothing to pool: the part is its own union, its group, instance and
        # lags included, which the pooled form drops (review 2026-10-05, YB8).
        return dict(parts[0])
    cols = _columns(parts[0])
    targets = list(parts[0].get("targets") or [])
    for p in parts[1:]:
        if _columns(p) != cols:
            msg = f"merge() needs the same columns in every part: {cols} vs {_columns(p)}"
            raise ValueError(msg)
        if list(p.get("targets") or []) != targets:
            msg = "merge() needs the same targets in every part"
            raise ValueError(msg)

    # Packed parts merge packed, entry for entry; any whole one makes the
    # merge whole. Float32 parts merge in float64.
    packed = all(_is_packed(np, p) for p in parts)
    iu = np.triu_indices(len(cols))

    def matrix(p: dict[str, Any]) -> Any:
        if packed:
            return np.asarray(p["comoments"], dtype=float)
        return _comoments(np, p)

    w = float(parts[0]["weight_sum"])
    mean = np.asarray(parts[0]["means"], dtype=float).copy()
    como = matrix(parts[0]).copy()
    q = _q_of(parts[0])
    tw = np.asarray(parts[0]["target_weights"], dtype=float).copy()
    cross = np.asarray(parts[0]["cross_moments"], dtype=float).copy()
    by_target = _opt(np, parts[0].get("means_by_target"))
    tmean = _opt(np, parts[0]["target_means"])
    tvar = _opt(np, parts[0]["target_vars"])
    tq = _target_q(np, parts[0])
    _, icept = _feature_slots(parts[0], None)
    cc = _opt(np, parts[0].get("cross_centred"))
    ybar = _ybar_of(np, parts[0], icept)

    for p in parts[1:]:
        wb = float(p["weight_sum"])
        total = w + wb
        if total > 0.0:
            mb = np.asarray(p["means"], dtype=float)
            d = mb - mean
            cb = matrix(p)
            spread = np.outer(d, d)[iu] if packed else np.outer(d, d)
            como = (w * como + wb * cb) / total + (w * wb / total**2) * spread
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
        # Each target's column means pool over its own rows, by its weights,
        # and its centred cross-moments as the co-moments do: each part's,
        # weighted, plus the product of the gaps between the parts' column
        # means and target means (review 2026-09-12, N4).
        by_target_b = _opt(np, p.get("means_by_target"))
        cc_b, ybar_b = _opt(np, p.get("cross_centred")), _ybar_of(np, p, icept)
        if any(v is None for v in (cc, cc_b, ybar, ybar_b, by_target, by_target_b)):
            cc = ybar = None
        elif cc.size or cc_b.size:
            a = np.divide(tw, ttotal, out=np.zeros_like(ttotal), where=live)
            b = np.divide(twb, ttotal, out=np.zeros_like(ttotal), where=live)
            dy = np.where(live, ybar_b - ybar, 0.0)
            dm = np.where(live[:, None], by_target_b - by_target, 0.0)
            cc = a[:, None] * cc + b[:, None] * cc_b + (a * b * dy)[:, None] * dm
            ybar = ybar + b * dy
        if by_target is None or by_target_b is None:
            by_target = None
        elif by_target.size or by_target_b.size:
            a = np.divide(tw, ttotal, out=np.zeros_like(ttotal), where=live)
            b = np.divide(twb, ttotal, out=np.zeros_like(ttotal), where=live)
            by_target = a[:, None] * by_target + b[:, None] * by_target_b
        tw = ttotal

    return {
        "group": None,
        "instance": None,
        "columns": cols,
        "targets": targets,
        "weight_sum": w,
        "n_kish": None if q is None or q <= 0.0 else w * w / q,
        "means": mean,
        "comoments": como,
        "cross_moments": cross,
        "means_by_target": by_target,
        "cross_centred": cc,
        "target_weights": tw,
        "target_means": tmean,
        "target_vars": tvar,
        "target_n_kish": None
        if tq is None
        else np.divide(tw * tw, tq, out=np.full_like(tq, np.nan), where=tq > 0.0),
        # A lagged cross-moment pairs a row with the row `l` back *in its own
        # part*, and the pairing across a part boundary is exactly what no
        # part holds. There is no Chan-style update for it, so a pooled Gram
        # reports none rather than a plausible wrong one (E56).
        "lags": None,
        "lag_comoments": None,
    }


def _row_mapping(row: Any) -> dict[str, Any]:
    """One closed row as a mapping, from a one-row frame, a row of
    ``iter_rows(named=True)``, or a mapping already."""
    to_dicts = getattr(row, "to_dicts", None)
    if to_dicts is not None:  # a polars DataFrame
        rows = to_dicts()
        if len(rows) != 1:
            msg = f"from_row() takes one closed row; this frame has {len(rows)}"
            raise ValueError(msg)
        return dict(rows[0])
    try:
        return dict(row)
    except (TypeError, ValueError) as e:
        msg = (
            "from_row() takes a one-row frame from ModelBank.closed_groups(), a row of its "
            f"iter_rows(named=True), or a mapping; got {type(row).__name__}"
        )
        raise TypeError(msg) from e


def _unvech(np: Any, flat: Any, k: int) -> Any:
    """The symmetric ``k x k`` matrix whose upper triangle, row by row, is
    ``flat`` -- the inverse of the closed row's packing."""
    v = np.asarray([np.nan if x is None else x for x in flat], dtype=float)
    want = k * (k + 1) // 2
    if v.size != want:
        msg = f"comoments has {v.size} entries; a {k}-column Gram packs {want}"
        raise ValueError(msg)
    m = np.zeros((k, k))
    iu = np.triu_indices(k)
    m[iu] = v
    return m + np.triu(m, 1).T


def _floats(np: Any, v: Any) -> Any:
    """A list column's values as floats, with null read as ``nan`` -- the
    frame writes null where the state says NaN (an undefined ``corr``, a
    target with no weighted row)."""
    return np.asarray([np.nan if x is None else x for x in v], dtype=float)


def from_row(row: Any) -> dict[str, Any]:
    """A closed group's row as the mapping :meth:`~polars_online.ModelBank.gram`
    returns, so everything in this module works on it.

    Takes a one-row frame from :meth:`~polars_online.ModelBank.closed_groups`, a
    row of its ``iter_rows(named=True)``, or a mapping. The row's ``comoments`` is
    the upper triangle with the diagonal, row by row, and its ``cross_moments``,
    ``means_by_target`` and ``cross_centred`` are row-major ``(n_targets, k)``;
    this expands all four. A closed group writes one row per Gram, so a row's
    ``targets`` are that Gram's.

    .. code-block:: python

        blocks = po.spec.ew_cov(
            "cov", features=["x0", "x1"], lam=1.0, group="block", group_close="monotone"
        )
        bank = po.ModelBank([blocks])
        bank.fit_predict(df.with_columns(block=pl.int_range(pl.len()) // 100))
        closed = bank.closed_groups()                 # one row per finished block
        for row in closed.iter_rows(named=True):
            g = po.gram.from_row(row)
            r = po.gram.correlation(g)

    The result is what ``gram()`` would have returned for that group bit for bit,
    except the co-moment matrix's lower triangle, which is the upper one mirrored.
    The two differ in the last bit or so and not more: the accumulator updates
    ``C[i][j]`` and ``C[j][i]`` with the same two products in the opposite order,
    which does not commute in IEEE arithmetic. Everything read off the matrix (a
    solve, a correlation, a condition number) is unaffected at that scale, and the
    packed half is what makes the closed row half the size.

    ``ValueError`` for a row of a kind that keeps no accumulators (its ``columns``
    is null): there is no Gram to make. ``TypeError`` for something that is none
    of the three forms.
    """
    np = _np()
    d = _row_mapping(row)
    cols = d.get("columns")
    if cols is None:
        name = d.get("spec", "this spec")
        msg = (
            f"closed row for {name!r} has no accumulators to make a Gram from; only "
            "ewridge, lasso and ew_cov keep a co-moment matrix"
        )
        raise ValueError(msg)
    columns = list(cols)
    k = len(columns)
    targets = list(d.get("targets") or [])
    cross = _floats(np, d.get("cross_moments") or [])
    n_kish = d.get("n_kish")
    tkish = d.get("target_n_kish")
    return {
        "group": d.get("group"),
        "instance": d.get("instance"),
        "columns": columns,
        "targets": targets,
        "weight_sum": float(d["weight_sum"]),
        "n_kish": None if n_kish is None else float(n_kish),
        "means": _floats(np, d["means"]),
        "comoments": _unvech(np, d["comoments"], k),
        "cross_moments": cross.reshape(len(targets), k) if targets else np.zeros((0, k)),
        "means_by_target": _floats(np, d["means_by_target"]).reshape(len(targets), k)
        if targets and d.get("means_by_target")
        else np.zeros((0, k)),
        "cross_centred": _floats(np, d["cross_centred"]).reshape(len(targets), k)
        if targets and d.get("cross_centred")
        else (None if targets else np.zeros((0, k))),
        "target_weights": _floats(np, d.get("target_weights") or []),
        "target_means": None if d.get("target_means") is None else _floats(np, d["target_means"]),
        "target_vars": None if d.get("target_vars") is None else _floats(np, d["target_vars"]),
        "target_n_kish": None if tkish is None else _floats(np, tkish),
        "lags": None if d.get("lags") is None else [int(v) for v in d["lags"]],
        "lag_comoments": None
        if d.get("lag_comoments") is None
        else _floats(np, d["lag_comoments"]).reshape(len(d["lags"]), k, k),
    }


def _opt(np: Any, v: Any) -> Any:
    return None if v is None else np.asarray(v, dtype=float).copy()


def _add_opt(a: Any, b: Any) -> Any:
    return None if b is None else a + b


def _q_of(g: dict[str, Any]) -> float | None:
    """`sum(w**2)` behind the feature moments, back out of `n_kish`: 0 for a
    part that learned nothing, whose `n_kish` is None for want of a weight
    to divide by, not for want of the sums (review round 4, YB6)."""
    w = float(g["weight_sum"])
    if w == 0.0:
        return 0.0
    nk = g.get("n_kish")
    if nk is None or not nk > 0.0:
        return None
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

    Exact, not approximate: a marginal set of moments is a sub-block of the joint
    ones, so this is a selection rather than a recomputation, and a regression on
    the subset is the regression the full accumulator implies. That is the point:
    forward stepwise, an information criterion over feature sets, or an
    ``r``-column fit read off a ``k``-column stream all fall out of one pass.

    ``cols`` are names or positions, and the intercept may be selected like any
    other column. Targets are untouched: they index a different axis. Returns a
    mapping of the same shape, its ``comoments`` in float64 and packed if
    ``g``'s is. ``KeyError`` for a name the Gram has not got,
    ``IndexError`` for a position.
    """
    np = _np()
    idx = _col_index(g, cols)
    names = _columns(g)
    como = _comoments(np, g)
    cross = np.asarray(g["cross_moments"], dtype=float)
    by_target = g.get("means_by_target")
    by_target = None if by_target is None else np.asarray(by_target, dtype=float)
    cc = g.get("cross_centred")
    cc = None if cc is None else np.asarray(cc, dtype=float)
    lag = g.get("lag_comoments")
    return {
        **g,
        "columns": [names[i] for i in idx],
        "means": np.asarray(g["means"], dtype=float)[idx],
        "comoments": como[np.ix_(idx, idx)][np.triu_indices(len(idx))]
        if _is_packed(np, g)
        else como[np.ix_(idx, idx)],
        "cross_moments": cross[:, idx] if cross.size else cross,
        "means_by_target": by_target[:, idx]
        if by_target is not None and by_target.ndim == 2
        else by_target,
        "cross_centred": cc[:, idx] if cc is not None and cc.ndim == 2 else cc,
        # The lagged matrices are over the same axes, so they slice the same
        # way -- in both, since a lagged matrix is not symmetric.
        "lag_comoments": None if lag is None else np.asarray(lag, dtype=float)[:, idx][:, :, idx],
    }


def correlation(g: dict[str, Any]) -> Any:
    """The correlation matrix of the columns, from the centred co-moments.

    A ``k x k`` array. ``nan`` in the row and column of a constant column, the
    intercept included: a constant has no correlation with anything, and reporting
    0 there would read as "independent". The diagonal is 1 where the variance is
    positive.
    """
    np = _np()
    c = _comoments(np, g)
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

    The model's own algebra (``EwRidge::solve``), so the result is the fit that
    spec would report on the same accumulator -- except under ``ridge_scale = "sum"`` or
    ``coef_prior``, whose penalties this does not take: a decaying prior on the
    sum scale that reaches the intercept, and a target other than zero. Their
    fits are not reproduced here. With an intercept in ``columns`` it
    is eliminated, and the slopes solve the centred system ``(C + ridge*I) b =
    c``: ``C`` the features' centred ``comoments``, and ``c = cross_centred[t]``,
    the target's cross-moments centred at its own column means ``m``
    (``means_by_target``) and its mean ``ybar``, as the model holds them; then
    ``b_0 = ybar - m . b``. That is the raw normal equations with the intercept
    unpenalized, exactly, where the target's rows are the Gram's, and it is what
    ``target_gaps = "pairwise"`` solves where they are not. A mapping without
    ``cross_centred`` forms ``cross_moments[t] - m * ybar`` instead, which at a
    level ``L`` keeps ``L**2 * eps`` of it. Without an intercept nothing is
    centred: ``(E[z z'] + ridge*I) b = E[z y]``, every slot penalized.

    .. rubric:: Parameters

    ``ridge``
        The penalty. A sequence gives one row of coefficients per value. The
        penalty is uniform in the basis being solved, so a grid rides a single
        eigendecomposition: with ``V d V'`` in hand every ridge is ``V diag(1/(d +
        r)) V' b``, which is what makes a grid of fifty cheap.
    ``target``
        The target, by name or position among this Gram's ``targets``.
    ``features``
        The regressors, narrowed: equivalent to :func:`subset` first. The
        intercept is refused here, since the solve handles it itself.
    ``standardize``
        Scale either system to correlation form, add ``ridge`` there and unscale,
        so ``ridge`` means the same thing whatever the features' units. A column
        with zero variance (zero raw second moment, without an intercept) is
        dropped with a coefficient of 0 rather than making the system singular.
        Pass the ``standardize`` the spec used, or the numbers will not match its
        ``coef()``.

    Returns a vector over the Gram's columns, starting with the intercept when
    there is one, in :func:`polars_online.spec.coef_index` order; with a sequence
    of ridges, one row per value.

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0,
            ridge=0.1, standardize=True,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("ridge")[0]
        beta = po.gram.solve(g, target="y", ridge=0.1, standardize=True)   # the spec's own fit
        grid = po.gram.solve(g, target="y", ridge=[1e-6, 0.01, 0.1, 1.0])  # one row per ridge

    ``KeyError`` / ``IndexError`` for a target or column the Gram has not got;
    ``ValueError`` for the intercept among ``features``, and for a ``ridge``
    that is not finite and at least 0, as a spec's ``ridge`` must be.
    """
    np = _np()
    ridges = np.atleast_1d(np.asarray(ridge, dtype=float))
    if not np.all(np.isfinite(ridges) & (ridges >= 0.0)):
        # A negative ridge solved a different, possibly indefinite, system
        # and returned it as the fit (review round 4, YB12).
        msg = f"solve: ridge must be finite and >= 0, got {ridge!r}"
        raise ValueError(msg)
    t = _target_index(g, target)
    slots, icept = _feature_slots(g, features)
    scalar = np.ndim(ridge) == 0
    k = len(_columns(g))

    means = np.asarray(g["means"], dtype=float)
    cross = np.asarray(g["cross_moments"], dtype=float)[t]
    como = _comoments(np, g)
    out = np.zeros((len(ridges), k))

    m, ybar = means, 0.0
    if icept >= 0:
        m = _means_of(np, g, t)
        ybar = cross[icept]
        a = como[np.ix_(slots, slots)]
        b = _cross_centred(np, g, t, m, ybar)[slots]
    else:
        a = como[np.ix_(slots, slots)] + np.outer(means[slots], means[slots])
        b = cross[slots]
    if standardize:
        s = np.sqrt(np.clip(np.diag(a), 0.0, None))
        keep = [i for i in range(len(slots)) if s[i] > 0.0]
    else:
        s = np.ones(len(slots))
        keep = list(range(len(slots)))
    if keep:
        kk = np.ix_(keep, keep)
        d, v = np.linalg.eigh(a[kk] / np.outer(s[keep], s[keep]))
        vb = v.T @ (b[keep] / s[keep])
        cols = [slots[i] for i in keep]
        for i, r in enumerate(ridges):
            out[i, cols] = (v @ (vb / (d + r))) / s[keep]
    if icept >= 0:
        out[:, icept] = ybar - out[:, slots] @ m[slots]
    return out[0] if scalar else out


@renamed_keywords("po.gram.lasso_path", {"lambdas": "penalties"})
def lasso_path(
    g: dict[str, Any],
    penalties: Sequence[float],
    *,
    l1_ratio: float = 1.0,
    penalty_weights: Sequence[float] | None = None,
    target: str | int = 0,
    features: Sequence[str | int] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> Any:
    """The elastic-net path from the Gram, one row of coefficients per penalty.

    The ``lasso`` model's coordinate descent (``Lasso::solve``), run offline on
    the standardized (correlation-form) matrix, warm-started down the path in the
    order given:

    .. code-block:: text

        b_i = soft(rho_i, l * l1_ratio * pw_i) / (C_ii + l * (1 - l1_ratio) * pw_i)
        soft(v, t) = sign(v) * max(|v| - t, 0)

    where ``rho_i`` is the standardized cross-correlation less the other columns'
    contributions. Coefficients come back in original units with the intercept
    recovered from the means, so a row is directly comparable to ``bank.coef()``.
    With an intercept the system is centred at the target's own means, as the
    model's descent reads it; through the origin nothing is centred and the raw
    moments are used, as the model does.

    The descent runs in Rust, about 70 times faster than the Python loop it
    replaced: on 40 penalties at 15 strongly correlated columns, 1.8 ms where
    that loop took 129 ms. Each sweep updates the coordinates in the same
    order, so a row matches that loop's to rounding (``2e-15`` there), not to
    the bit: the sum over the other columns runs in a different order. For
    the start of a path, knot by knot and exactly, see :func:`lars_path`.

    .. rubric:: Parameters

    ``penalties``
        The penalties ``l``, from large to small, as a path is meant to be
        walked: the warm start makes that both faster and better conditioned.
        They are a spec's ``lasso_path``, and ``bank.coef()`` reports each as
        ``penalty``. The keyword was ``lambdas`` before 1.0; that name is
        refused, and the error names this one.
    ``l1_ratio``
        The share of the penalty that is L1; below 1 an elastic net. Default 1.
    ``penalty_weights``
        A scale on the penalty per feature, in ``features`` order (or ``columns``
        order without the intercept): 0 leaves a column unpenalized, and a column
        the stream found constant is dropped whatever is asked for. The online
        model has no such parameter; it is the one thing here the models do not
        also do, and it is cheap offline because the path is re-walked rather than
        carried.
    ``target``, ``features``
        As for :func:`solve`.
    ``max_iter``, ``tol``
        The model's ``max_iter`` and ``tol``.

    Returns an array of shape ``(len(penalties), k)`` over the Gram's columns, the
    intercept first when there is one.

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("ridge")[0]
        path = po.gram.lasso_path(g, [0.1, 0.01, 0.001], target="y")   # one row per penalty

    ``ValueError`` for ``penalty_weights`` of the wrong length, and for a
    number outside its range, by the rule a spec applies to the same
    parameter: a penalty or a weight that is not finite and at least 0,
    ``l1_ratio`` outside ``[0, 1]``, ``max_iter`` below 1, ``tol`` not finite
    and above 0. ``KeyError`` / ``IndexError`` as for :func:`solve`.
    """
    np = _np()
    # Each was taken as given: `max_iter=0` returned the intercept with every
    # slope 0, a fully penalised fit to look at, and `tol=nan` never stopped
    # early (review round 4, YB12).
    lams = np.asarray(penalties, dtype=float).reshape(-1)
    if not np.all(np.isfinite(lams) & (lams >= 0.0)):
        msg = f"lasso_path: penalties must be finite and >= 0, got {list(penalties)!r}"
        raise ValueError(msg)
    if not 0.0 <= l1_ratio <= 1.0:
        msg = f"lasso_path: l1_ratio must be in [0, 1], got {l1_ratio!r}"
        raise ValueError(msg)
    if max_iter < 1:
        msg = (
            f"lasso_path: max_iter must be >= 1, got {max_iter!r}: it is the coordinate "
            "descent's sweeps, and with none every slope stays at 0"
        )
        raise ValueError(msg)
    if not (math.isfinite(tol) and tol > 0.0):
        msg = f"lasso_path: tol must be finite and > 0, got {tol!r}"
        raise ValueError(msg)
    t = _target_index(g, target)
    slots, icept = _feature_slots(g, features)
    k = len(_columns(g))
    pw = (
        np.ones(len(slots)) if penalty_weights is None else np.asarray(penalty_weights, dtype=float)
    )
    if pw.shape != (len(slots),):
        msg = f"penalty_weights must have one entry per feature ({len(slots)}), got {pw.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(pw) & (pw >= 0.0)):
        msg = f"lasso_path: penalty_weights must be finite and >= 0, got {pw.tolist()!r}"
        raise ValueError(msg)
    flat = _native.gram_cd_path(
        _native_gram(np, g, icept),
        t,
        slots,
        None if icept < 0 else icept,
        [float(v) for v in lams],
        float(l1_ratio),
        [float(v) for v in pw],
        int(max_iter),
        float(tol),
    )
    return np.asarray(flat, dtype=float).reshape(len(lams), k)


def _native_gram(np: Any, g: dict[str, Any], icept: int) -> tuple[Any, ...]:
    """A Gram's arrays as the Rust fits read them: ``k`` and
    ``weight_sum``, then ``means``, ``comoments``, ``cross_moments``,
    ``means_by_target``, ``cross_centred``, ``target_weights``,
    ``target_means``, ``target_vars`` and ``target_n_kish``, each
    C-contiguous float64. A mapping without ``means_by_target`` or
    ``cross_centred`` gets them as :func:`solve` forms them: ``means`` for
    every target, and ``cross_moments[t] - m * ybar``; one without a target
    moment gets ``nan`` for it."""
    k = len(_columns(g))
    means = np.ascontiguousarray(g["means"], dtype=np.float64)
    cross = np.ascontiguousarray(np.asarray(g["cross_moments"], dtype=np.float64).reshape(-1, k))
    m = cross.shape[0]
    by_target = g.get("means_by_target")
    if by_target is None or len(by_target) == 0:
        by_target = np.tile(means, (m, 1))
    by_target = np.ascontiguousarray(np.asarray(by_target, dtype=np.float64).reshape(m, k))
    cc = g.get("cross_centred")
    if cc is None or (len(cc) == 0 and m > 0):
        cc = np.zeros((m, k))
        if icept >= 0:
            for t in range(m):
                cc[t] = _cross_centred(np, g, t, by_target[t], cross[t][icept])
    cc = np.ascontiguousarray(np.asarray(cc, dtype=np.float64).reshape(m, k))
    # The co-moments go over as they are, float32 or packed included: the
    # Rust side reads every form, and a copy here would be the Gram's size.
    como = np.asarray(g["comoments"])
    if como.dtype not in (np.float64, np.float32):
        como = como.astype(np.float64)
    como = np.ascontiguousarray(como.reshape(-1))

    def per_target(key: str) -> Any:
        v = g.get(key)
        if v is None:
            return np.full(m, np.nan)
        return np.ascontiguousarray(v, dtype=np.float64).reshape(m)

    return (
        k,
        float(g["weight_sum"]),
        means,
        como,
        cross,
        by_target,
        cc,
        per_target("target_weights"),
        per_target("target_means"),
        per_target("target_vars"),
        per_target("target_n_kish"),
    )


def _limit(name: str, v: int | None) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        msg = f"{name} must be a whole number >= 1 or None, got {v!r}"
        raise ValueError(msg)
    return v


def _lars_weights(np: Any, n: int, penalty_weights: Sequence[float] | None) -> list[float]:
    if penalty_weights is None:
        return []
    pw = np.asarray(penalty_weights, dtype=float)
    if pw.shape != (n,):
        msg = f"penalty_weights must have one entry per feature ({n}), got {pw.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(pw) & (pw > 0.0)):
        msg = (
            f"lars_path: penalty_weights must be finite and > 0, got {pw.tolist()!r}; a weight "
            "of 0 leaves a column unpenalized, in the fit at every penalty, which is no point a "
            "path from the largest penalty can start at: use lasso_path for it"
        )
        raise ValueError(msg)
    return [float(v) for v in pw]


def lars_path(
    g: dict[str, Any],
    *,
    target: str | int = 0,
    features: Sequence[str | int] | None = None,
    max_steps: int | None = None,
    max_active: int | None = None,
    penalty_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The lasso path from its start, knot by knot, by least angle regression.

    Efron, Hastie, Johnstone and Tibshirani's LARS (2004) with their lasso
    modification, on the correlation form :func:`lasso_path` standardizes
    the Gram to: ``R`` the features' correlation matrix and ``d`` their
    standardized cross-moments with the target, centred at the target's own
    means where there is an intercept. The lasso minimizes

    .. code-block:: text

        f(b) = 0.5 * b' R b - d' b + l * sum_j w_j * |b_j|

    and its solution is piecewise linear in the penalty ``l``. The path
    starts at ``l_max = max_j |d_j| / w_j``, where every slope is 0. Each
    knot is where a column enters the active set ``A``, or leaves it as its
    slope crosses 0. Between knots, each active column's correlation with
    the residual, ``c_j = d_j - (R b)_j``, stays at ``l * w_j`` in size:

    .. code-block:: text

        delta_A = inv(R_AA) (w_A * s_A)       s_j = sign(c_j)
        b_A    += gamma * delta_A
        l      -= gamma

    The step ``gamma`` is the first at which a column outside ``A`` reaches
    the active correlation, an active slope reaches 0, or ``l`` reaches 0.
    At ``l = 0`` the fit is least squares on ``A``. A knot's coefficients
    are the lasso's exactly at its penalty, so ``lasso_path(g, [l])`` at a
    knot's ``l`` gives the same row where its descent has converged. Its
    ``tol`` bounds a sweep's largest move, not the distance from the
    solution: at a condition number of 1e8 a descent at ``tol=1e-14``
    stopped 0.48 from a knot in the intercept, at an objective above the
    knot's.

    A selection reads only the start of a path, and a step costs
    ``O(k |A|)``, so stop it early: after ``max_steps`` knots past the
    first, or at the knot where ``max_active`` columns are active. A column
    that is constant on the Gram's rows never enters. A column that would
    enter collinear with the active set is set aside, since it adds no
    direction: its pivot in the factor of ``R_AA`` is below ``1e-12`` of
    its variance. It takes no step and makes no knot, so each knot but the
    last at ``l = 0`` is a column entering or leaving. Ties go to the
    column first in ``features``, which is the Gram's ``columns`` order
    when ``features`` is ``None``. Measured on
    an M4 Pro, a path takes 9 microseconds whole at 15 columns, and 0.2 ms
    for 40 knots at 200 columns or 3 ms at 2,000.

    .. rubric:: Parameters

    ``target``, ``features``
        As for :func:`solve`.
    ``max_steps``, ``max_active``
        Where to stop: knots past the first, and columns active at once.
        ``None`` is no limit, and the path runs to ``l = 0``.
    ``penalty_weights``
        A scale ``w_j`` on the penalty per feature, as for
        :func:`lasso_path`, but each finite and above 0. A weight of 0
        leaves a column unpenalized, in the fit at every penalty, which is
        no point a path from ``l_max`` can start at. Use :func:`lasso_path`
        for that, and for an elastic net, whose path is not piecewise
        linear.

    Returns a dict:

    ``penalties``
        The penalty ``l`` at each knot, falling, in the units of
        :func:`lasso_path` and of the ``lasso`` model.
    ``coef``
        Shape ``(n_knots, k)``: each knot's coefficients over the Gram's
        columns, in the features' original units with the intercept
        recovered, as a row of :func:`lasso_path`.
    ``active``
        The active set below each knot, by name, in the order its columns
        entered. A column entering at a knot is in that knot's set with its
        coefficient still 0, growing from there; one leaving is out of it.
        So the first knot's set holds the first column to enter, and a
        knot's nonzero coefficients are the set before it, less any column
        leaving at it.
    ``stop``
        ``"max_steps"``, ``"max_active"``, or ``"end"``: ``l`` reached 0,
        or no column can enter.

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("ridge")[0]
        path = po.gram.lars_path(g, target="y", max_active=1)
        first = path["active"][-1]              # [the column that enters first]

    ``ValueError`` for a limit that is not a whole number of at least 1, and
    for ``penalty_weights`` of the wrong length or not finite and above 0.
    ``KeyError`` / ``IndexError`` as for :func:`solve`.
    """
    return lars_paths(
        [g],
        targets=[target],
        features=features,
        max_steps=max_steps,
        max_active=max_active,
        penalty_weights=penalty_weights,
    )[0][0]


def lars_paths(
    grams: Sequence[dict[str, Any]],
    *,
    targets: Sequence[str | int] | None = None,
    features: Sequence[str | int] | None = None,
    max_steps: int | None = None,
    max_active: int | None = None,
    penalty_weights: Sequence[float] | None = None,
) -> list[list[dict[str, Any]]]:
    """:func:`lars_path` for many Grams and targets at once, on the bank's
    thread pool.

    Returns one list per Gram, with one path per target. The targets are
    every target of each Gram when ``targets`` is ``None``, else those
    named, by name or position, in each. Each Gram's correlation matrix is
    formed once, and its targets' paths run beside each other on it. The
    Grams run beside each other too, on the pool ``POLARS_ONLINE_MAX_THREADS``
    sizes. A path is the same arithmetic on one thread whatever the pool's
    size, so its numbers are those of :func:`lars_path` to the bit. The
    other parameters are those of :func:`lars_path`, applied to every path.

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                               half_life=100.0, group="stock_id")
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        paths = po.gram.lars_paths(bank.gram("ridge"), max_active=1)   # one list per group
        firsts = [per_gram[0]["active"][-1] for per_gram in paths]

    Every Gram must have the same ``columns`` (``ValueError``), as for
    :func:`merge`. Otherwise it raises as :func:`lars_path` does.
    """
    np = _np()
    parts = list(grams)
    if not parts:
        return []
    cols = _columns(parts[0])
    for p in parts[1:]:
        if _columns(p) != cols:
            msg = f"lars_paths() needs the same columns in every Gram: {cols} vs {_columns(p)}"
            raise ValueError(msg)
    slots, icept = _feature_slots(parts[0], features)
    steps, active = _limit("max_steps", max_steps), _limit("max_active", max_active)
    weights = _lars_weights(np, len(slots), penalty_weights)
    tidx = [
        list(range(len(p["cross_moments"])))
        if targets is None
        else [_target_index(p, t) for t in targets]
        for p in parts
    ]
    native = [_native_gram(np, p, icept) for p in parts]
    out = _native.gram_lars_paths(
        native, tidx, slots, None if icept < 0 else icept, weights, steps, active
    )
    k = len(cols)
    return [
        [
            {
                "penalties": np.asarray(pen, dtype=float),
                "coef": np.frombuffer(coef, dtype=np.float64).reshape(len(pen), k),
                "active": [[cols[i] for i in a] for a in act],
                "stop": stop,
            }
            for pen, coef, act, stop in per
        ]
        for per in out
    ]


_PATH_KEYS = ("max_steps", "max_active", "penalty_weights")


def solve_subsets(
    grams: Sequence[dict[str, Any]],
    subsets: Sequence[Sequence[int]],
    *,
    targets: Sequence[str | int] | None = None,
    features: Sequence[str | int] | None = None,
    ridge: float = 0.0,
    standardize: bool = False,
    path: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    """Merge each subset of the Grams and fit it, for many subsets at once.

    For each subset, a list of positions in ``grams``, the parts are pooled
    as :func:`merge` pools them and each target is fitted on the pooled
    Gram: a ridge with its statistics, as :func:`solve` and then
    :func:`coef_stats` give them, or with ``path`` its lasso path, as
    :func:`lars_path` gives it. Two uses: the halves of a stability
    selection, each a merge of about half the blocks; and each block's own
    fit, its ``t`` saying how stable a coefficient is across blocks.

    The Grams are read once, into memory of the call's own, and every
    subset is merged and fitted there, on the bank's thread pool, without a
    Python copy of a matrix per subset. Measured on an M4 Pro under a load
    of 12, for 38 Grams and 100 subsets of 19: 0.02 s at 200 columns and
    3.0 s at 2,000, where the loop of :func:`merge`, :func:`solve` and
    :func:`coef_stats` took 0.43 s and 76 s. The copy is the Grams' size
    again for the call's length. A subset's merge is formed on the
    slots and the intercept alone, and dropped when its fits are done; its
    targets share one factorization. The merge is :func:`merge`'s
    arithmetic entry for entry, so it agrees with it to the bit. The fit
    then is

    .. code-block:: text

        (A + ridge * I) b = r_t            A, r_t as solve() forms them
        b_0       = ybar_t - m_t . b       the intercept, with one
        resid_var = Var[y] - 2 b' cov_xy + b' C b
        sigma2    = resid_var * n / (n - p)      n = target_n_kish, p the columns
        se        = sqrt(diag(inv(C)) * sigma2 / n),   t = b / se

    with ``C`` the slots' centred co-moments. Under ``standardize`` the
    system is solved in correlation form, as :func:`solve` solves it. Both
    systems are factorized by Cholesky, where :func:`solve` runs numpy's
    symmetric eigendecomposition and :func:`coef_stats` its inverse, so the
    two agree to rounding, the system's condition number times ``1e-16``:
    measured within ``5e-13`` of each number's size at 2,000 columns. A
    system whose factorization fails, such as a constant column under a
    ridge of 0, gives ``nan`` here, where :func:`solve` divides by an
    eigenvalue of 0 or of rounding size. A nearly singular one gives what
    any solve of it gives, rounding noise.

    .. rubric:: Parameters

    ``grams``
        The Grams, each as :meth:`~polars_online.ModelBank.gram` returns
        it, all with the same ``columns`` and ``targets``.
    ``subsets``
        One list of positions in ``grams`` per fit, each naming a Gram at
        most once: a Gram twice in one merge counts its rows twice. ``[[0],
        [1], ...]`` fits each Gram alone.
    ``targets``
        The targets to fit, by name or position: every target when
        ``None``.
    ``features``
        As for :func:`solve`.
    ``ridge``, ``standardize``
        As for :func:`solve`, one ridge.
    ``path``
        ``None`` for the ridge fit, or a dict of :func:`lars_path`'s
        ``max_steps``, ``max_active`` and ``penalty_weights`` (``{}`` for
        none of them) for each target's lasso path instead. ``ridge`` and
        ``standardize`` are then not read: a path is in correlation form.

    Returns one list per subset, with one dict per target. A ridge fit's
    dict has ``coef``, the coefficients over the Gram's columns, and
    :func:`coef_stats`' ``resid_var``, ``sigma2``, ``r2``, ``n``, ``se`` and
    ``t``. A path's dict is :func:`lars_path`'s.

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                               half_life=float("inf"), group="stock_id")
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        blocks = bank.gram("ridge")                     # one Gram per group
        each = po.gram.solve_subsets(blocks, [[i] for i in range(len(blocks))], ridge=1e-6)
        t_x0 = [fits[0]["t"][1] for fits in each]       # x0's t in every block
        halves = po.gram.solve_subsets(blocks, [[0, 1], [2, 3]], path={"max_active": 1})

    ``ValueError`` for Grams of different columns or targets, an empty
    subset, a position out of range or named twice in one subset, a
    ``ridge`` that is not finite and at least 0, an unknown key in
    ``path``, and, for a ridge fit, a Gram without target moments (as
    :func:`coef_stats`) or one without ``means_by_target`` and
    ``cross_centred``, which :func:`merge` forms after pooling and this
    does not. Otherwise as :func:`solve` and :func:`lars_path`.
    """
    np = _np()
    parts = list(grams)
    if not parts:
        msg = "solve_subsets() needs at least one Gram"
        raise ValueError(msg)
    cols = _columns(parts[0])
    names = list(parts[0].get("targets") or [])
    for p in parts[1:]:
        if _columns(p) != cols or list(p.get("targets") or []) != names:
            msg = "solve_subsets() needs the same columns and targets in every Gram"
            raise ValueError(msg)
    picked: list[list[int]] = []
    for s in subsets:
        idx = [int(i) for i in s]
        if not idx:
            msg = "solve_subsets(): a subset names no Gram"
            raise ValueError(msg)
        if any(not 0 <= i < len(parts) for i in idx):
            msg = f"solve_subsets(): subset {list(s)!r} names a Gram out of range 0..{len(parts)}"
            raise ValueError(msg)
        if len(set(idx)) != len(idx):
            msg = (
                f"solve_subsets(): subset {list(s)!r} names a Gram twice, which would count "
                "its rows twice"
            )
            raise ValueError(msg)
        picked.append(idx)
    slots, icept = _feature_slots(parts[0], features)
    m = len(parts[0]["target_weights"])
    tidx = list(range(m)) if targets is None else [_target_index(parts[0], t) for t in targets]
    for p in parts:
        for key in ("means_by_target", "cross_centred"):
            v = p.get(key)
            if m and (v is None or len(v) == 0):
                msg = (
                    f"solve_subsets() needs {key!r} in every Gram, which ModelBank.gram() "
                    "reports; merge() then solve() forms it from cross_moments instead"
                )
                raise ValueError(msg)
    native = [_native_gram(np, p, icept) for p in parts]
    ic = None if icept < 0 else icept
    k = len(cols)
    if path is not None:
        unknown = sorted(set(path) - set(_PATH_KEYS))
        if unknown:
            msg = f"solve_subsets(): path takes {list(_PATH_KEYS)}, not {unknown}"
            raise ValueError(msg)
        steps = _limit("max_steps", path.get("max_steps"))
        active = _limit("max_active", path.get("max_active"))
        weights = _lars_weights(np, len(slots), path.get("penalty_weights"))
        out = _native.gram_path_subsets(native, picked, tidx, slots, ic, weights, steps, active)
        return [
            [
                {
                    "penalties": np.asarray(pen, dtype=float),
                    "coef": np.frombuffer(coef, dtype=np.float64).reshape(len(pen), k),
                    "active": [[cols[i] for i in a] for a in act],
                    "stop": stop,
                }
                for pen, coef, act, stop in per
            ]
            for per in out
        ]
    if not (math.isfinite(ridge) and ridge >= 0.0):
        msg = f"solve_subsets: ridge must be finite and >= 0, got {ridge!r}"
        raise ValueError(msg)
    for p in parts:
        if p.get("target_vars") is None or p.get("target_n_kish") is None:
            msg = (
                "a Gram here has no target moments, so it has no Var[y] to take a residual "
                "variance from; a state saved by 0.2.0 or earlier reports None for them"
            )
            raise ValueError(msg)
    fits = _native.gram_ridge_subsets(
        native, picked, tidx, slots, ic, float(ridge), bool(standardize)
    )
    return [
        [
            {
                "coef": np.frombuffer(coef, dtype=np.float64),
                "resid_var": resid_var,
                "sigma2": sigma2,
                "r2": r2,
                "n": n,
                "se": np.frombuffer(se, dtype=np.float64),
                "t": np.frombuffer(t, dtype=np.float64),
            }
            for coef, se, t, resid_var, sigma2, r2, n in per
        ]
        for per in fits
    ]


def coef_stats(
    g: dict[str, Any],
    coef: Sequence[float],
    *,
    target: str | int = 0,
    features: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    """Residual variance, standard errors and t-statistics for a fit.

    This is what the target moments were added for: with ``Var[y]`` in the Gram, a
    saved state answers "how good is this fit, and which coefficients are real"
    without the rows:

    .. code-block:: text

        resid_var = Var[y] - 2 b' Cov[X, y] + b' C b
        sigma2    = resid_var * n / (n - k)          # n = target_n_kish
        se        = sqrt(diag(inv(C)) * sigma2 / n)
        t         = b / se

    Returns a dict:

    ``resid_var``, ``sigma2``, ``r2``
        The residual variance of the fit, its degrees-of-freedom-corrected
        estimate, and ``1 - resid_var / Var[y]``.
    ``n``
        The Kish size the correction and the errors use.
    ``se``, ``t``
        Arrays over the same slots as ``coef``, with the intercept's entry
        ``nan``: its standard error depends on the design's centring, which the
        Gram has already absorbed.

    ``n`` is Kish's effective sample size, not ``weight_sum``: a weighted stream's
    weight sum is not a count, and dividing by it would report standard errors too
    small by the factor the weights are unequal by. The rows behind an
    exponentially weighted fit are also neither independent nor identically
    distributed, so read a ``t`` here as a scale for comparing coefficients, not
    as a p-value.

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0,
            ridge=0.1, standardize=True,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("ridge")[0]
        beta = po.gram.solve(g, target="y", ridge=0.1, standardize=True)
        stats = po.gram.coef_stats(g, beta, target="y")    # resid_var, sigma2, r2, n, se, t

    ``coef`` must have one entry per Gram column (``ValueError``); ``ValueError``
    if the Gram has no target moments, which a state saved by 0.2.0 or earlier
    cannot answer.
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
    cross = np.asarray(g["cross_moments"], dtype=float)[t]
    como = _comoments(np, g)
    c = como[np.ix_(slots, slots)]
    var_y = float(np.asarray(g["target_vars"], dtype=float)[t])
    n = float(np.asarray(g["target_n_kish"], dtype=float)[t])
    ybar = cross[icept] if icept >= 0 else float(np.asarray(g["target_means"], dtype=float)[t])
    # Centred cross-covariance over the target's rows, the pair `comoments`
    # is in, as the model holds it (N4).
    cov_xy = _cross_centred(np, g, t, _means_of(np, g, t), ybar)[slots]

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
    """Variance inflation factors: ``1 / (1 - R2_j)`` for each column on the rest,
    straight off the diagonal of the inverse correlation matrix.

    An array over ``features`` (the Gram's columns without the intercept by
    default: a constant is perfectly explained by any other constant, so its VIF
    is undefined). A column the stream found constant reports ``inf``. Above
    about 10 the coefficient of a column is mostly noise; the fix is a
    ridge, a subset, or a feature set the spec already knows how to fit
    beside the full one.

    The diagonal is read from the eigendecomposition ``R = V diag(d) V'`` as
    ``VIF_j = sum_i V_ji^2 / d_i``. An eigenvalue at or below numpy's rank
    tolerance, ``d_max * k * eps``, is a dependency: a column with a weight
    past ``sqrt(eps)`` in its direction reports ``inf``. A pseudo-inverse
    drops that direction instead and read an exact duplicate at 0.25, below
    the floor of 1 a VIF cannot go under (docs/PLAN.md task 223).

    So a column that is an exact linear combination of the others, where
    ``R2_j`` is 1, reads ``inf`` or a number past 1e10, not always ``inf``.
    Rounding in the accumulated moments can leave the dependency's
    eigenvalue just above the tolerance, and the column then reads about
    1e14. A copy times 100 plus 3 read 2.2e14 at ``lam = 0.999`` and
    ``inf`` at ``lam = 1``; three columns where the third is the sum of the
    first two read 1.2e14 to 2.4e14 at ``lam = 1`` and ``inf`` at 0.999
    (5,000 rows). Read anything past 1e10 as a dependency. A column whose
    part in a dependency is tiny reads near 1, as if it stood apart: in
    ``c = a + b`` with ``a`` spread a million times as wide as ``b``, ``b``'s
    weight in the dependency's direction, in correlation units, is 5e-13,
    far below ``sqrt(eps)``, and it read 1.0002 where ``a`` and ``c`` read
    ``inf``. A large VIF says a column is in a dependency; a VIF near 1 does
    not say it is in none.
    """
    np = _np()
    slots, _ = _feature_slots(g, features)
    r = correlation(g)[np.ix_(slots, slots)]
    out = np.full(len(slots), np.inf)
    # A constant column's correlations are NaN across its row and its
    # column, so its diagonal says which it is: a whole row's finiteness
    # left out every column beside it (review 6, C-4).
    ok = [i for i in range(len(slots)) if np.isfinite(r[i, i])]
    if not ok:
        return out
    d, v = np.linalg.eigh(r[np.ix_(ok, ok)])
    null = d <= d.max() * len(ok) * np.finfo(float).eps
    w = v**2
    inflated = (w[:, ~null] / d[~null]).sum(axis=1)
    inflated[w[:, null].sum(axis=1) > math.sqrt(np.finfo(float).eps)] = np.inf
    out[ok] = inflated
    return out


def condition(g: dict[str, Any], *, features: Sequence[str | int] | None = None) -> dict[str, Any]:
    """Belsley's collinearity diagnostics for the accumulated design.

    Returns a dict:

    ``columns``
        The columns diagnosed, in order.
    ``singular_values``
        Of the column-scaled design, largest first.
    ``condition_indexes``
        ``s_max / s_j`` for each.
    ``kappa``
        The largest of them.
    ``proportions``
        The variance-decomposition proportions: one row per component and one
        column per feature, each column summing to 1.

    A component with a large condition index and a large share of two or more
    columns' variance is a near-dependency between exactly those columns, which is
    what makes this worth more than a single ``kappa``: it says which columns are
    the problem, where a VIF only says that one is. Belsley's rule of thumb is an
    index above 30 with two proportions above 0.5. The design is scaled to unit
    column length first, Belsley's prescription, but not centred: the intercept is
    part of the collinearity when a column is nearly constant, and centring hides
    that. ``singular_values`` are of that scaled raw matrix, so they are the
    square roots of the eigenvalues of the scaled second-moment matrix.
    """
    np = _np()
    names = _columns(g)
    slots = list(range(len(names))) if features is None else _col_index(g, features)
    means = np.asarray(g["means"], dtype=float)
    como = _comoments(np, g)
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
