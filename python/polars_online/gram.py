"""Pool, narrow, solve and diagnose the Grams a bank exports.

A Gram is the set of sums behind a least-squares fit: the weight of the
rows, the columns' means and co-moments, and each target's moments.
:meth:`~polars_online.ModelBank.gram` returns one for each group and each
instance of a spec (the part of a model that fits one half-life), from one
pass over data that is never held in memory.
This module works on them after the pass, in five kinds of call:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - what you want
     - the calls
   * - shape a Gram
     - :func:`merge` pools shards into their union, or within each part;
       :func:`subset` keeps some of the columns; :func:`from_row` reads a
       closed group's row
   * - fit it
     - :func:`solve` for a ridge; :func:`lasso_path` for the elastic net
       over a grid of penalties; :func:`lars_path` and :func:`lars_paths`
       for the lasso path from its start, knot by knot
   * - judge a fit
     - :func:`coef_stats`: residual variance, standard errors and ``t``
   * - many fits at once
     - :func:`solve_subsets` merges each subset of the Grams and fits it
   * - diagnose the design
     - :func:`correlation`, :func:`vif` and :func:`condition`

Every function but :func:`from_row` takes the mappings ``gram()`` returns,
one or a list, described by :class:`Gram`. Between them, the functions
read its ``columns``, ``targets``, ``means``, ``comoments``,
``lag_comoments``, ``cross_moments``, ``means_by_target``,
``cross_centered``, ``target_weights``, ``target_means``, ``target_vars``,
``weight_sum``, ``n_kish`` and ``target_n_kish``. :func:`merge`, :func:`subset` and
:func:`from_row` return a mapping of the same shape, so a closed group's
row (:meth:`~polars_online.ModelBank.closed_groups`) is read the same way.
The compact forms ``gram()`` returns under ``dtype="float32"`` or
``layout="packed"`` are taken too, and read in float64. :func:`merge` with
``within=True`` returns two more keys, ``within_parts`` and
``target_within_parts``, which every fit reads.

The arithmetic is the models' own, so :func:`solve` on a spec's Gram
reproduces that spec's coefficients where it takes the spec's penalty, and
:func:`lasso_path` reproduces the ``lasso`` model's path. It does not agree
to the last bit. The models factorize with ``faer``'s Cholesky. Here numpy
runs LAPACK's symmetric eigendecomposition (``numpy.linalg.eigh``) in
:func:`solve` and :func:`vif`, and an inverse (``numpy.linalg.inv``) in
:func:`coef_stats`. Where the columns' scales differ, :func:`solve` takes
an LU solve for each ridge above 0 (its Notes say when). Each of these
rounds differently in the last place or two, so the tests hold the two to
a relative tolerance, not to equality.
The paths and :func:`solve_subsets` run in Rust. :func:`lars_paths` and
:func:`solve_subsets` run on the process's threads, or on as many as their
``threads`` gives.

The module needs numpy, an optional extra of this package (``pip install
polars-online[numpy]``) and not a dependency, since it is not one of
Polars' either. Without it, every function raises ``ModuleNotFoundError``.
Nothing here needs scipy or scikit-learn.

`docs/GRAM.md <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__ is the guide:
the theory behind each function, and a table from each task on weekly blocks from one pass to
the how-to whose recipe runs it.
"""

from __future__ import annotations

import math
import numbers
import operator
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from polars_online import _polars_online as _native
from polars_online._bank import _check_threads
from polars_online._errors import WrongTypeError
from polars_online._frame import ReadinessWarning, _user_stacklevel
from polars_online._renamed import renamed_keywords
from polars_online._warnings import deprecated_keywords, forward_deprecated

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "Gram",
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


class Gram(TypedDict):
    """The mapping that :meth:`~polars_online.ModelBank.gram` returns and the functions here take.

    It holds the Gram behind one fit as numpy arrays: the weight of the rows,
    the columns' means and co-moments, and each target's moments. Use it as
    the type of a Gram you pass between the functions of this module.
    :meth:`~polars_online.ModelBank.gram` says what each key holds and its
    shape.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("ridge")[0]                  # the spec has no groups: one Gram
        print(g["columns"], g["targets"], g["comoments"].shape)

    .. code-block:: text

        ['intercept', 'x0', 'x1'] ['y'] (3, 3)

    .. rubric:: See also

    - :meth:`~polars_online.ModelBank.gram`, which returns it and describes
      each key.
    - :func:`from_row`, which builds one from a closed group's row.

    .. rubric:: Notes

    :func:`merge`, :func:`subset` and :func:`from_row` return the same keys,
    so a Gram from any of them goes on to any other function here. The
    arrays are typed ``Any`` because numpy is an optional extra of this
    package. ``within_parts`` and ``target_within_parts`` are present only on
    a Gram pooled within parts (``merge(within=True)``).
    """

    group: str | None
    instance: str | None
    columns: list[str]
    targets: list[str]
    weight_sum: float
    n_kish: float | None
    means: Any
    comoments: Any
    cross_moments: Any
    means_by_target: Any
    cross_centered: Any
    target_weights: Any
    target_means: Any
    target_vars: Any
    target_n_kish: Any
    lags: list[int] | None
    lag_comoments: Any
    within_parts: NotRequired[Any]
    target_within_parts: NotRequired[Any]


def _np() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as e:  # pragma: no cover - exercised by a stub
        msg = (
            "po.gram: it works in numpy arrays, but numpy is not installed. "
            "Install it with `pip install numpy` or `pip install polars-online[numpy]`."
        )
        raise ModuleNotFoundError(msg) from e
    return np


def _columns(g: Gram, who: str = "po.gram") -> list[str]:
    cols = g.get("columns")
    if cols is None:
        msg = (
            f"{who}: a Gram must name its columns, but this mapping has no 'columns' key. "
            "Pass a Gram that ModelBank.gram() or po.gram.from_row returns."
        )
        raise KeyError(msg)
    return list(cols)


def _as_position(v: Any) -> int | None:
    """``v`` as an integer -- anything ``operator.index`` reads as one,
    numpy's integers included -- or ``None`` for a name, a bool, or anything
    else, which is looked up as a name (task 269, D-6)."""
    if isinstance(v, (str, bool)):
        return None
    try:
        return operator.index(v)
    except TypeError:
        return None


def _col_index(g: Gram, cols: Sequence[str | int], who: str = "po.gram") -> list[int]:
    names = _columns(g, who)
    # Each name's first position, looked up once: `list.index` per name made
    # a subset of 400 of 1,500 columns take 3 ms (task 256).
    where: dict[str, int] = {}
    for i, name in enumerate(names):
        where.setdefault(name, i)
    out = []
    for c in cols:
        at = _as_position(c)
        if at is not None:
            if not -len(names) <= at < len(names):
                msg = (
                    f"{who}: a column's position must be from {-len(names)} to {len(names) - 1} "
                    "for a "
                    f"Gram of {len(names)} columns, but it is {c}. Name the column, or give "
                    "a position inside the Gram."
                )
                raise IndexError(msg)
            out.append(at % len(names))
        elif isinstance(c, str) and c in where:
            out.append(where[c])
        else:
            msg = (
                f"{who}: the Gram has no column {c!r}, but it has {_listed(names)}. Name one of "
                "the "
                "Gram's columns."
            )
            raise KeyError(msg)
    return out


def _target_index(g: Gram, target: str | int, who: str = "po.gram") -> int:
    names = list(g.get("targets") or [])
    i = _as_position(target)
    if i is not None:
        n = len(g["cross_moments"])
        if n == 0:
            msg = (
                f"{who}: the Gram has no targets, but target is {target}. Pass the Gram of a "
                "spec with targets, such as ewridge or lasso."
            )
            raise IndexError(msg)
        if not -n <= i < n:
            msg = (
                f"{who}: a target's position must be from {-n} to {n - 1} for a Gram of {n} "
                f"targets, but it is {target}. Name the target, or give a position inside the "
                "Gram."
            )
            raise IndexError(msg)
        return i % n
    if not isinstance(target, str) or target not in names:
        if not names:
            msg = (
                f"{who}: the Gram has no targets, but target is {target!r}. Pass the Gram of a "
                "spec with targets, such as ewridge or lasso."
            )
            raise KeyError(msg)
        msg = (
            f"{who}: the Gram has no target {target!r}, but it has {_listed(names)}. Name one of "
            "the Gram's targets."
        )
        raise KeyError(msg)
    return names.index(target)


def _listed(names: Sequence[Any]) -> str:
    """Names for a message: the list whole up to ten, else its first ten and
    the count of the rest."""
    names = list(names)
    if len(names) <= 10:
        return repr(names)
    return f"{names[:10]!r} and {len(names) - 10} more"


def _repeated(names: list[str], slots: list[int]) -> str | None:
    """The name of the first column ``slots`` holds twice, or ``None``."""
    seen: set[int] = set()
    for c in slots:
        if c in seen:
            return names[c]
        seen.add(c)
    return None


def _refuse_repeated(names: list[str], slots: list[int], who: str, key: str) -> None:
    """``ValueError`` when ``slots`` names a column twice, by name or by
    position, or by both: its moments would be read twice, and the matrix
    of the two copies is singular."""
    twice = _repeated(names, slots)
    if twice is not None:
        msg = (
            f"{who}: {key} names column {twice!r} twice, and two copies of one column make a "
            "singular matrix. Name each column once."
        )
        raise ValueError(msg)


def _means_of(np: Any, g: Gram, t: int) -> Any:
    """The column means target ``t``'s cross-moments are centred at and its
    intercept is recovered from: its own, over the rows it was present on
    (``means_by_target``), or the Gram's ``means`` in a mapping without
    them."""
    by_target = g.get("means_by_target")
    if by_target is None or len(by_target) == 0:
        return np.asarray(g["means"], dtype=float)
    return np.asarray(by_target, dtype=float)[t]


def _cross_centered(np: Any, g: Gram, t: int, m: Any, ybar: float) -> Any:
    """Target ``t``'s cross-moments centred at its column means ``m`` and its
    mean ``ybar``, as the model holds them: ``cross_centered``, or -- in a
    mapping without it -- ``cross_moments[t] - m * ybar``, a difference of
    two numbers the size of ``L**2`` at a level ``L``, which keeps ``L**2 *
    eps`` of the answer."""
    cc = g.get("cross_centered")
    if cc is not None and len(cc):
        return np.asarray(cc, dtype=float)[t]
    return np.asarray(g["cross_moments"], dtype=float)[t] - m * ybar


def _ybar_of(np: Any, g: Gram, icept: int) -> Any:
    """Each target's mean, which its centred cross-moments are centred at:
    the raw cross-moment at the intercept, exactly (``0 + 1 * ybar``), or
    ``target_means``; ``None`` when the mapping has neither."""
    cross = np.asarray(g["cross_moments"], dtype=float)
    if icept >= 0 and cross.size:
        return cross[:, icept].copy()
    tm = g.get("target_means")
    return None if tm is None else np.asarray(tm, dtype=float).copy()


def _feature_slots(
    g: Gram, features: Sequence[str | int] | None, who: str = "po.gram"
) -> tuple[list[int], int]:
    """The column positions to regress on, and the intercept's position or -1.

    The intercept is never one of the features: it is a constant column with
    zero variance, and treating it as a regressor is how a solve ends up
    singular.
    """
    names = _columns(g, who)
    icept = names.index(INTERCEPT) if INTERCEPT in names else -1
    if features is None:
        slots = [i for i in range(len(names)) if i != icept]
    else:
        slots = _col_index(g, features, who)
        if icept in slots:
            msg = (
                f"{who}: features names {INTERCEPT!r}, but it is the intercept's constant "
                "column, which the fit adds itself. Leave it out of features."
            )
            raise ValueError(msg)
        twice = _repeated(names, slots)
        if twice is not None:
            msg = (
                f"{who}: features names column {twice!r} twice, and a fit given one column "
                "twice has no single answer, so one coefficient would overwrite the other. "
                "Name each column once."
            )
            raise ValueError(msg)
    return slots, icept


def _is_packed(np: Any, g: Gram) -> bool:
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


def _comoments(np: Any, g: Gram) -> Any:
    """``g``'s co-moments as the whole ``k x k`` matrix in float64, whatever
    form ``ModelBank.gram()`` handed them over in (``dtype``, ``layout``)."""
    if _is_packed(np, g):
        return _unpack(np, g["comoments"], len(_columns(g)))
    return np.asarray(g["comoments"], dtype=float)


def merge(grams: Sequence[Gram], *, within: bool = False) -> Gram:
    """Pool the Grams of disjoint sets of rows into the Gram of their union, or within each part.

    Use it to combine Grams that share a weighting: one per shard of a pass,
    one per group, one per worker, or one per week of a stream. To keep some
    of a Gram's columns, use :func:`subset`. To merge many subsets and fit
    each in one call, use :func:`solve_subsets`.

    .. rubric:: Parameters

    ``grams``
        The parts, each as :meth:`~polars_online.ModelBank.gram` returns it,
        all with the same ``columns`` and ``targets``. Merging one Gram
        without ``within`` returns it unchanged. Parts in a compact form
        (``ModelBank.gram(dtype="float32")`` or ``layout="packed"``) merge in
        float64. Under ``target_gaps="own_rows"`` a spec may have a Gram per
        set of targets: merge the entries with the same ``targets`` across
        the shards.
    ``within``
        ``False``, the default, pools the parts into the Gram of their
        union. ``True`` pools each part's moments about its own means and
        leaves out the gaps between the parts' means: the Gram of the rows
        demeaned part by part, whose fit has an intercept per part (the
        within, or fixed-effects, estimator). Notes give the arithmetic of
        both.

    .. rubric:: Returns

    A mapping of the same shape as each part (:class:`Gram`). When every
    part is packed, the merge is packed too, entry for entry; otherwise it
    is the whole matrix. Some keys come back ``None``:

    - ``lags`` and ``lag_comoments``: a lagged cross-moment pairs a row with
      the row ``l`` back within its own part, and no part holds the pairs
      across a part boundary.
    - ``group`` and ``instance``: a pooled Gram belongs to no one group or
      instance (the part of a model that fits one half-life).
    - ``n_kish``, where a part has none, such as a part under a hard window
      whose sum of squared weights kept no digit
      (:meth:`~polars_online.ModelBank.gram`): the sum behind it is not
      there to add. A part that learned nothing is the one case apart: it
      has no ``n_kish`` and adds nothing to the sum.
    - the target moments, where a part has none, such as a mapping built by
      hand without them.
    - ``means_by_target`` and ``cross_centered``, where a part has none.
      :func:`solve` then forms them from ``cross_moments``.

    With ``within=True`` it has two keys more: ``within_parts`` counts the
    parts with weight, and ``target_within_parts`` counts, per target, the
    parts with weight for it. A part already pooled within adds its own
    counts. Every fit reads them: :func:`solve`, :func:`lasso_path`,
    :func:`lars_path` and :func:`solve_subsets` give the slopes and ``nan``
    for the intercept, and :func:`coef_stats` takes one degree of freedom
    per part in place of the intercept's one.

    .. rubric:: Raises

    ``ValueError``
        When ``grams`` is empty, when the parts differ in ``columns`` or
        ``targets``, or when, without ``within``, several parts include one
        pooled within parts. Such a part's means are 0, so a union would add
        the gap from 0 to the other parts' means, as if its rows had not
        been demeaned. With ``within=True``, also when a part has no
        ``cross_centered`` and neither an intercept column nor
        ``target_means`` to center its cross-moments at.
    ``KeyError``
        When a part has no ``columns`` key, such as a mapping built by hand.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0, group="stock_id"
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        parts = bank.gram("ridge")                      # one Gram per stock
        pooled = po.gram.merge(parts)                   # the Gram of every stock's rows together
        within = po.gram.merge(parts, within=True)      # each stock's level left out
        print(len(parts), round(pooled["weight_sum"], 2), within["within_parts"])

    .. code-block:: text

        4 289.54 4

    .. rubric:: See also

    - :func:`solve_subsets`, which merges many subsets of the Grams and fits
      each.
    - :func:`subset`, which keeps some of a Gram's columns.
    - *Pooling blocks* in `docs/GRAM.md
      <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__,
      the theory, and `Pool Grams
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/pool-grams.md>`__,
      a recipe for pooling, for aging a block first, and for pooling within
      blocks.

    .. rubric:: Notes

    **The union** follows Chan, Golub and LeVeque: the pooled
    co-moments are the weighted average of the parts' plus the spread
    between their means. With weights ``W_a`` and ``W_b``, means ``m_a`` and
    ``m_b``, co-moments ``C_a`` and ``C_b``, squared-weight sums ``Q_a`` and
    ``Q_b``, and the gap between the means ``d = m_b - m_a``:

    .. code-block:: text

        W = W_a + W_b
        m = m_a + (W_b / W) * d
        C = (W_a * C_a + W_b * C_b) / W + (W_a * W_b / W**2) * outer(d, d)
        Q = Q_a + Q_b

    Each target's ``means_by_target`` pools over its own rows, by its
    ``target_weights``, and its ``cross_centered`` as the co-moments do, with
    the gaps between the parts' column means and target means.

    **Each part enters centered at its own means**, never as a raw sum that
    a mean is subtracted from afterwards, so data far from zero keep their
    digits. The parts are folded in one at a time, so the rounding grows
    with their number. It was measured on 100,000 rows of three columns at a
    level of 1e4 standard deviations, against the exact union in rational
    arithmetic, relative to the spread:

    .. list-table::
       :header-rows: 1
       :widths: 60 40

       * - how the union was formed
         - the co-moments' error
       * - this merge of 2 parts, each exact to float64
         - 2.4e-15
       * - this merge of 1,000 parts
         - 3.8e-14
       * - raw sums of 2 parts, centered afterwards
         - 4.0e-08
       * - raw sums of 1,000 parts, centered afterwards
         - 2.5e-07
       * - one bank's own Gram, learned over all the rows
         - 3.8e-14

    **Pooling within parts.** For parts ``i`` with weights ``W_i``,
    co-moments ``C_i``, and for each target its weights ``T_i``, centered
    cross-moments ``c_i`` and variances ``V_i``:

    .. code-block:: text

        W = sum W_i          C = sum W_i C_i / W
        T = sum T_i          c = sum T_i c_i / T          V = sum T_i V_i / T
        n_kish = W**2 / sum(W_i**2 / n_kish_i),   target_n_kish likewise

    folded in the parts' order as the union is, so where the parts' means
    are equal the two forms give the same numbers. Each feature is less its
    part's mean, over the target's rows for that target's moments, and each
    target less its part's mean. So the result's ``means`` and
    ``means_by_target`` are 0, the intercept's 1 aside, its
    ``cross_moments`` are its ``cross_centered``, and its ``target_means``
    are 0. Wooldridge (2010, chapter 10) calls a regression on it the
    within, or fixed-effects, estimator. Its slopes say how the columns and
    the target move together inside each part, and the parts' levels drop
    out. A part with no weight for a target (``T_i = 0``) adds to ``C`` and
    not to that target's moments.

    **To merge two halves of a decayed stream in time order, decay the
    earlier half to the later half's last row first.** Halves run under
    ``half_life=float("inf")`` merge as they are. Under a finite half-life,
    each half's weights are relative to its own last row, so the earlier
    half is over-weighted by exactly the decay between them. Multiply the earlier half's
    ``weight_sum`` and ``target_weights`` by ``0.5 ** (dt / half_life)``,
    with ``dt`` the clock from its last row to the later half's last row,
    each step capped at ``gap_cap`` as the bank caps it. Leave every other
    key as it is. The means and co-moments are weighted means, and
    ``n_kish`` and ``target_n_kish`` do not change when every weight is
    scaled, so the sums of squared weights the merge recovers from them
    decay with the weights. For example, this code reads ``df`` from the
    README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        halves = po.spec.ewridge("h", targets=["y"], features=["x0", "x1"],
                                 clock="t", gap_cap=10.0, half_life=100.0)

        def gram_of(rows):                               # the Gram of a new bank fed these rows
            fitted = po.ModelBank([halves])
            fitted.fit_predict(rows)
            return fitted.gram("h")[0]

        # Each half's Gram, the early one aged to the late one's last row.
        early, late = gram_of(df.head(200)), gram_of(df.tail(200))
        dt = df["t"][399] - df["t"][199]                 # early half's last row to late half's
        decay = 0.5 ** (dt / 100.0)
        early = dict(early, weight_sum=early["weight_sum"] * decay,
                     target_weights=early["target_weights"] * decay)
        aged = po.gram.merge([early, late])              # the Gram of one bank over all 400 rows
        whole = gram_of(df)
        print(np.allclose(aged["comoments"], whole["comoments"], rtol=1e-12, atol=1e-12),
              round(float(aged["weight_sum"]), 6), round(float(whole["weight_sum"]), 6))

    .. code-block:: text

        True 135.721952 135.721952
    """
    np = _np()
    parts = list(grams)
    if not parts:
        msg = (
            "po.gram.merge: grams must hold at least one Gram, but it is empty. Pass the Grams "
            "to pool."
        )
        raise ValueError(msg)
    if len(parts) == 1 and not within:
        # Nothing to pool: the part is its own union, its group, instance and
        # lags included, which the pooled form drops (review 2026-10-05, YB8).
        return parts[0].copy()
    who = "po.gram.merge"
    cols = _columns(parts[0], who)
    targets = list(parts[0].get("targets") or [])
    for i, p in enumerate(parts[1:], 1):
        if _columns(p, who) != cols:
            msg = (
                f"{who}: every Gram must have the same columns, but Gram 0 has {_listed(cols)} "
                f"and Gram {i} has {_listed(_columns(p, who))}. Pass Grams of specs with the "
                "same features."
            )
            raise ValueError(msg)
        if list(p.get("targets") or []) != targets:
            msg = (
                f"{who}: every Gram must have the same targets, but Gram 0 has {targets} and "
                f"Gram {i} has {list(p.get('targets') or [])}. Pass Grams of specs with the "
                "same targets."
            )
            raise ValueError(msg)
    if within:
        return _merge_within(np, parts, cols, targets)
    _refuse_within_union(who, parts)

    # Packed parts merge packed, entry for entry; any whole one makes the
    # merge whole. Float32 parts merge in float64.
    packed = all(_is_packed(np, p) for p in parts)
    iu = np.triu_indices(len(cols))

    def matrix(p: Gram) -> Any:
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
    cc = _opt(np, parts[0].get("cross_centered"))
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
        cc_b, ybar_b = _opt(np, p.get("cross_centered")), _ybar_of(np, p, icept)
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
        "cross_centered": cc,
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


def _is_within(g: Gram) -> bool:
    """Whether ``g`` was pooled within parts (``merge(within=True)``)."""
    return g.get("within_parts") is not None


def _refuse_within_union(what: str, parts: Sequence[Gram]) -> None:
    """Refuse to pool a Gram pooled within parts as a union beside others:
    its means are 0, the demeaned rows', so the union would add the gap
    from 0 to the other parts' means as if their rows had not been
    demeaned."""
    for i, p in enumerate(parts):
        if _is_within(p):
            msg = (
                f"{what}: Gram {i} was pooled within parts (within_parts={p['within_parts']}), "
                "so its means are 0, and pooling it as a union would add the gap from 0 to the "
                "other Grams' means as if its rows had not been centered. Pass within=True to "
                "pool every Gram within parts."
            )
            raise ValueError(msg)


def _target_parts(np: Any, g: Gram) -> Any:
    """Per target, the parts ``g`` holds: its ``target_within_parts`` if it
    was pooled within, else 1 for each target with weight and 0 for one
    without."""
    if _is_within(g):
        return np.asarray(g["target_within_parts"], dtype=np.int64)
    return (np.asarray(g["target_weights"], dtype=float) > 0.0).astype(np.int64)


def _centred_rows(np: Any, g: Gram, icept: int, m: int, k: int) -> Any:
    """Every target's centred cross-moments, ``m x k``: ``cross_centered``,
    or in a mapping without it, as :func:`solve` forms them."""
    cc = g.get("cross_centered")
    if cc is not None and len(cc):
        return np.asarray(cc, dtype=float).reshape(m, k).copy()
    if m == 0:
        return np.zeros((0, k))
    ybar = _ybar_of(np, g, icept)
    if ybar is None:
        msg = (
            "po.gram.merge: within=True needs each target's cross-moments about its means, but "
            "a Gram has no cross_centered, and no intercept column or target_means to center "
            "them at. Pass the Grams as ModelBank.gram() returns them, which carry "
            "cross_centered."
        )
        raise ValueError(msg)
    return np.stack([_cross_centered(np, g, t, _means_of(np, g, t), ybar[t]) for t in range(m)])


def _merge_within(np: Any, parts: list[Gram], cols: list[str], targets: list[str]) -> Gram:
    """:func:`merge` with ``within=True``: the parts' moments about their
    own means, folded in order, and the Gram of the rows demeaned part by
    part around them. The fold is
    ``online_core::gramfit::merge_within``'s, entry for entry, so
    :func:`solve_subsets` agrees with it to the bit."""
    packed = all(_is_packed(np, p) for p in parts)
    k = len(cols)
    _, icept = _feature_slots(parts[0], None)

    def matrix(p: Gram) -> Any:
        if packed:
            return np.asarray(p["comoments"], dtype=float)
        return _comoments(np, p)

    w = float(parts[0]["weight_sum"])
    como = matrix(parts[0]).copy()
    q = _q_of(parts[0])
    for p in parts[1:]:
        wb = float(p["weight_sum"])
        total = w + wb
        if total > 0.0:
            como = (w * como + wb * matrix(p)) / total
        w = total
        q = None if q is None else _add_opt(q, _q_of(p))

    tw = np.asarray(parts[0]["target_weights"], dtype=float).copy()
    m = len(tw)
    tvar = _opt(np, parts[0]["target_vars"])
    tq = _target_q(np, parts[0])
    cc = _centred_rows(np, parts[0], icept, m, k)
    for p in parts[1:]:
        twb = np.asarray(p["target_weights"], dtype=float)
        ttotal = tw + twb
        live = ttotal > 0.0
        a = np.divide(tw, ttotal, out=np.zeros_like(ttotal), where=live)
        b = np.divide(twb, ttotal, out=np.zeros_like(ttotal), where=live)
        tvb = _opt(np, p["target_vars"])
        tvar = None if tvar is None or tvb is None else a * tvar + b * tvb
        tq = None if tq is None else _add_opt(tq, _target_q(np, p))
        cc = a[:, None] * cc + b[:, None] * _centred_rows(np, p, icept, m, k)
        tw = ttotal

    # The demeaned rows' means: 0 for every feature, and the intercept's
    # constant 1.
    means = np.zeros(k)
    if icept >= 0:
        means[icept] = 1.0
    return {
        "group": None,
        "instance": None,
        "columns": cols,
        "targets": targets,
        "weight_sum": w,
        "n_kish": None if q is None or q <= 0.0 else w * w / q,
        "means": means,
        "comoments": como,
        "cross_moments": cc.copy(),
        "means_by_target": np.tile(means, (m, 1)),
        "cross_centered": cc,
        "target_weights": tw,
        "target_means": np.zeros(m),
        "target_vars": tvar,
        "target_n_kish": None
        if tq is None
        else np.divide(tw * tw, tq, out=np.full_like(tq, np.nan), where=tq > 0.0),
        "lags": None,
        "lag_comoments": None,
        "within_parts": sum(
            int(p["within_parts"]) if _is_within(p) else int(float(p["weight_sum"]) > 0.0)
            for p in parts
        ),
        "target_within_parts": sum(
            (_target_parts(np, p) for p in parts[1:]), _target_parts(np, parts[0])
        ),
    }


def _row_mapping(row: Any) -> dict[str, Any]:
    """One closed row as a mapping, from a one-row frame, a row of
    ``iter_rows(named=True)``, or a mapping already."""
    to_dicts = getattr(row, "to_dicts", None)
    if to_dicts is not None:  # a polars DataFrame
        rows = to_dicts()
        if len(rows) != 1:
            msg = (
                f"po.gram.from_row: row must be one closed group's row, but the frame has "
                f"{len(rows)} rows. Pass one row, such as closed.row(i, named=True) or "
                "closed[i]."
            )
            raise ValueError(msg)
        return dict(rows[0])
    try:
        return dict(row)
    except (TypeError, ValueError) as e:
        msg = (
            "po.gram.from_row: row must be a one-row frame of ModelBank.closed_groups(), a "
            f"row of its iter_rows(named=True), or a mapping, but its type is "
            f"{type(row).__name__}. Pass one of those."
        )
        raise TypeError(msg) from e


def _unvech(np: Any, flat: Any, k: int) -> Any:
    """The symmetric ``k x k`` matrix whose upper triangle, row by row, is
    ``flat`` -- the inverse of the closed row's packing."""
    v = np.asarray([np.nan if x is None else x for x in flat], dtype=float)
    want = k * (k + 1) // 2
    if v.size != want:
        msg = (
            f"po.gram.from_row: comoments must hold {want} entries, the upper triangle of a "
            f"Gram of {k} columns, but it holds {v.size}. Pass the row as "
            "ModelBank.closed_groups() wrote it."
        )
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


def from_row(row: Any) -> Gram:
    """Read a closed group's row as the Gram :meth:`~polars_online.ModelBank.gram` would return.

    Use it on the rows :meth:`~polars_online.ModelBank.closed_groups` writes,
    or on those rows read back from a file, so that every function in this
    module works on them. For a group still open in the bank, use
    :meth:`~polars_online.ModelBank.gram` itself.

    .. rubric:: Parameters

    ``row``
        One closed group's row: a one-row frame of
        :meth:`~polars_online.ModelBank.closed_groups`, a row of its
        ``iter_rows(named=True)``, or a mapping. A closed group writes one
        row per Gram, so a row's ``targets`` are the targets of that one
        Gram. A row that names ``cross_centered`` by its old British spelling,
        ``cross_centred``, is read with a warning (see Raises).

    .. rubric:: Returns

    A :class:`Gram`. The row's ``comoments`` is the upper triangle with the
    diagonal, row by row, and its ``cross_moments``, ``means_by_target`` and
    ``cross_centered`` are row-major ``(n_targets, k)``, with ``k`` the
    number of columns. This function expands all four.

    .. rubric:: Raises

    ``ValueError``
        When the row is of a model that keeps no Gram (its ``columns`` is
        null), when a frame holds more or fewer than one row, or when
        ``comoments`` does not hold ``k (k + 1) / 2`` entries.
    ``TypeError``
        When ``row`` is none of the three forms, or names both
        ``cross_centered`` and ``cross_centred``.
    :class:`~polars_online.PolarsOnlineDeprecationWarning`
        A warning, not an error: issued when the row names
        ``cross_centred``. That spelling is read until the next major
        version.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        blocks = po.spec.ew_cov(
            "cov", features=["x0", "x1"], lam=1.0, group="block", group_close="monotone"
        )
        bank = po.ModelBank([blocks])
        bank.fit_predict(df.with_columns(block=pl.int_range(pl.len()) // 100))
        closed = bank.closed_groups()                 # one row per closed block
        for row in closed.iter_rows(named=True):
            g = po.gram.from_row(row)
            r = po.gram.correlation(g)
            print(row["group"], round(float(r[0, 1]), 3))

    .. code-block:: text

        0 0.021
        1 0.108
        2 -0.053

    .. rubric:: See also

    - :meth:`~polars_online.ModelBank.closed_groups`, which writes the rows.
    - `Read Grams from saved files
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/read-grams-from-saved-files.md>`__,
      a recipe that runs, with numpy alone too.

    .. rubric:: Notes

    The result is what ``gram()`` would have returned for that group, bit
    for bit. The model keeps one triangle of its co-moment matrix and reads
    the other through its mirror, so the upper triangle the row holds loses
    nothing, and keeping only that triangle makes the row about half the
    size of the whole matrix.
    """
    np = _np()
    d = forward_deprecated(
        "po.gram.from_row", _row_mapping(row), {"cross_centred": "cross_centered"}
    )
    cols = d.get("columns")
    if cols is None:
        name = d.get("spec")
        whose = "this closed group" if name is None else f"spec {name!r}'s closed group"
        msg = (
            f"po.gram.from_row: {whose} has no Gram, since only ewridge, lasso and ew_cov keep "
            "one. Pass a row of ModelBank.closed_groups() from one of those specs."
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
        "cross_centered": _floats(np, d["cross_centered"]).reshape(len(targets), k)
        if targets and d.get("cross_centered")
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


def _q_of(g: Gram) -> float | None:
    """`sum(w**2)` behind the feature moments, back out of `n_kish`: 0 for a
    part that learned nothing, whose `n_kish` is None for want of a weight
    to divide by, not for want of the sums."""
    w = float(g["weight_sum"])
    if w == 0.0:
        return 0.0
    nk = g.get("n_kish")
    if nk is None or not nk > 0.0:
        return None
    return w * w / float(nk)


def _target_q(np: Any, g: Gram) -> Any:
    nk = g.get("target_n_kish")
    if nk is None:
        return None
    nk = np.asarray(nk, dtype=float)
    tw = np.asarray(g["target_weights"], dtype=float)
    return np.divide(tw * tw, nk, out=np.zeros_like(tw), where=np.isfinite(nk) & (nk > 0.0))


@deprecated_keywords("po.gram.subset", {"cols": "columns"})
def subset(g: Gram, columns: Sequence[str | int]) -> Gram:
    """Keep some of a Gram's columns, in the order given.

    Use it to fit or diagnose a narrower design from a Gram that holds more
    columns: forward stepwise selection, an information criterion over
    feature sets, or an ``r``-column fit read off a ``k``-column stream, all
    from one pass. To narrow one fit, :func:`solve`, :func:`lasso_path`,
    :func:`lars_path` and :func:`coef_stats` take ``features`` instead.

    .. rubric:: Parameters

    ``g``
        The Gram, as :meth:`~polars_online.ModelBank.gram` returns it.
    ``columns``
        The columns to keep, by name or by position, as the Gram's own
        ``columns`` key names them. A negative position counts from the
        end. The intercept may be kept like any other column. **Leave the
        intercept out, and every fit on the subset goes through the
        origin**, as a spec with ``fit_intercept=False`` does: :func:`solve`
        and the paths then read the raw moments, ``E[z z']`` and ``E[z y]``,
        and center nothing. Keep it for a fit with an intercept. The
        keyword's older name, ``cols``, is still read, with a
        :class:`~polars_online.PolarsOnlineDeprecationWarning`.

    .. rubric:: Returns

    A :class:`Gram` of the columns kept: its ``columns``, ``means``,
    ``comoments``, ``cross_moments``, ``means_by_target``,
    ``cross_centered`` and ``lag_comoments`` narrowed to them, its
    ``comoments`` in float64, packed if ``g``'s is. The targets are
    untouched, since they index a different axis.

    .. rubric:: Raises

    ``KeyError``
        A name the Gram does not have.
    ``IndexError``
        A position outside the Gram's columns.
    ``TypeError``
        ``columns`` and its older name ``cols`` are given together.
    ``ValueError``
        A column named twice, by name or by position, since two copies of
        one column make the subset's matrix singular.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1", "x2"], half_life=100.0
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("ridge")[0]
        narrow = po.gram.subset(g, ["intercept", "x0"])     # the fit on x0 alone
        print(narrow["columns"], po.gram.solve(narrow).round(2))

    .. code-block:: text

        ['intercept', 'x0'] [1.03 1.81]

    .. rubric:: See also

    - :func:`solve`, whose ``features`` narrows one fit without a new Gram.
    - *Every least-squares fit is a function of the Gram* in `docs/GRAM.md
      <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__,
      the theory and a recipe that runs.

    .. rubric:: Notes

    The subset is exact. The moments of some of the columns are a
    sub-block of the moments of all of them, so this selects that block,
    and a regression on the subset is the regression the whole Gram implies
    for those columns.
    """
    np = _np()
    idx = _col_index(g, columns, "po.gram.subset")
    names = _columns(g, "po.gram.subset")
    _refuse_repeated(names, idx, "po.gram.subset", "columns")
    como = _comoments(np, g)
    cross = np.asarray(g["cross_moments"], dtype=float)
    by_target = g.get("means_by_target")
    by_target = None if by_target is None else np.asarray(by_target, dtype=float)
    cc = g.get("cross_centered")
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
        "cross_centered": cc[:, idx] if cc is not None and cc.ndim == 2 else cc,
        # The lagged matrices are over the same axes, so they slice the same
        # way -- in both, since a lagged matrix is not symmetric.
        "lag_comoments": None if lag is None else np.asarray(lag, dtype=float)[:, idx][:, :, idx],
    }


def correlation(g: Gram) -> Any:
    """Compute the correlation matrix of a Gram's columns from their co-moments.

    Use it to read how a Gram's columns move together, or to pass a
    correlation matrix on to :mod:`polars_online.corr`, whose functions read
    a Gram this way themselves. To judge whether a fit on the columns is
    determined, use :func:`vif` or :func:`condition`.

    .. rubric:: Parameters

    ``g``
        The Gram, as :meth:`~polars_online.ModelBank.gram` returns it, in
        any of its forms.

    .. rubric:: Returns

    A ``k x k`` float array, with ``k`` the number of columns:
    ``C_ij / sqrt(C_ii C_jj)`` from the co-moments ``C``. Its entries are
    ``nan`` in the row and the column of a constant column, the intercept
    included: a constant has no correlation with anything, and reporting 0
    there would read as "independent". The diagonal is 1, to rounding,
    where the variance is above 0.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ew_cov("cov", features=["x0", "x1", "x2"], half_life=100.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("cov")[0]
        print(g["columns"])
        print(po.gram.correlation(g).round(2))

    .. code-block:: text

        ['x0', 'x1', 'x2']
        [[ 1.    0.16  0.08]
         [ 0.16  1.   -0.01]
         [ 0.08 -0.01  1.  ]]

    .. rubric:: See also

    - :func:`vif` and :func:`condition`, which diagnose the same design.
    - :mod:`polars_online.corr`, which repairs, summarizes and scores a
      correlation matrix.
    - *Is the fit determined?* in `docs/GRAM.md
      <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__,
      the theory, and `Check that a fit is determined
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/check-a-fit-is-determined.md>`__,
      a recipe that runs.

    .. rubric:: Notes

    The diagonal is ``C_ii / sqrt(C_ii)**2``, which came within 2.2e-16 of
    1 on 200 random matrices.
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
    g: Gram,
    *,
    ridge: float | Sequence[float] = 0.0,
    target: str | int = 0,
    features: Sequence[str | int] | None = None,
    standardize: bool = False,
) -> Any:
    """Compute ridge coefficients from a Gram, in the features' original units.

    Use it to fit a ridge, or least squares at ``ridge=0``, on a Gram after
    the pass: the spec's own fit, a grid of penalties, or a fit on some of
    its columns. For the lasso or the elastic net, use :func:`lasso_path` or
    :func:`lars_path`. To fit many merged subsets of Grams at once, use
    :func:`solve_subsets`.

    .. rubric:: Parameters

    ``g``
        The Gram, as :meth:`~polars_online.ModelBank.gram` returns it, or as
        :func:`merge`, :func:`subset` or :func:`from_row` return one.
    ``ridge``
        The penalty, added to the co-moments, which are weighted means. It
        is a spec's ``ridge`` under its default ``ridge_scale="mean"``, in
        the features' squared units unless ``standardize``. A sequence gives
        one row of coefficients per value. Default 0, least squares.
        Refused unless finite and at least 0, as a spec's ``ridge`` is.
    ``target``
        The target, by name or by position among the Gram's ``targets``.
        Default 0, the first.
    ``features``
        The columns to fit on, by name or position: the fit on a
        :func:`subset` of these columns and the intercept, returned over all
        the Gram's columns with 0 for each column left out. Default
        ``None``, every column. The intercept is refused here, since the
        solve handles it itself, and so is a column named twice.
    ``standardize``
        ``True`` scales the system to correlation form, adds ``ridge`` there
        and scales back, so ``ridge`` means the same thing whatever the
        features' units. A column with no variance (no raw second moment,
        without an intercept) is then dropped with a coefficient of 0,
        where it would make the system singular. Pass the ``standardize``
        the spec used, or the numbers will not match its ``coef()``.
        Default ``False``.

    .. rubric:: Returns

    A float array over the Gram's columns, the intercept first when there
    is one, in :func:`polars_online.spec.coef_index` order. With a sequence
    of ridges, a 2-d array, one row per value. On a Gram pooled within parts
    (``merge(within=True)``) the slopes are the within estimator's and the
    intercept is ``nan``: each part had an intercept of its own, and the
    pooled Gram holds none of them.

    .. rubric:: Raises

    ``KeyError``
        A target or column name the Gram does not have, or a target named
        for a Gram with no targets.
    ``IndexError``
        A target or column position outside the Gram's, or any target
        position, the default ``0`` included, on a Gram with no targets.
    ``ValueError``
        The intercept among ``features``, or a column named twice in them
        (one column solved twice is a singular system, and only one of its
        two coefficients could be reported); a ``ridge`` that is not finite
        and at least 0.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0,
            ridge=0.1, standardize=True,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("ridge")[0]
        coef = po.gram.solve(g, target="y", ridge=0.1, standardize=True)   # the spec's own fit
        grid = po.gram.solve(g, target="y", ridge=[1e-6, 0.01, 0.1, 1.0])  # one row per ridge
        print(coef.round(3), grid.shape)

    .. code-block:: text

        [ 0.986  1.773 -0.884] (4, 3)

    .. rubric:: See also

    - :func:`coef_stats`, the standard errors and ``t`` of the fit.
    - :func:`lasso_path` and :func:`lars_path`, the lasso's fits.
    - *A ridge and its t* in `docs/GRAM.md
      <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__,
      the theory, and `Fit a ridge from a Gram
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/fit-a-ridge-from-a-gram.md>`__,
      a recipe that runs.

    .. rubric:: Notes

    **The arithmetic is the model's own**: the ``ewridge`` model's solve, so
    the result is the fit that spec would report on the same Gram. Two penalties are not
    taken here, and a spec with either reports a fit this does not
    reproduce: ``ridge_scale="sum"``, a decaying prior on the sum scale that
    reaches the intercept, and ``coef_prior``, a prior centered on values
    other than 0.

    With an intercept among the columns, it is eliminated, and the slopes
    ``b`` solve the centered system:

    .. code-block:: text

        (C + ridge * I) b = c
        b_0 = ybar - m . b

    ``C`` is the features' centered ``comoments``. ``c = cross_centered[t]``
    is target ``t``'s cross-moments centered at its own column means ``m``
    (``means_by_target``) and its mean ``ybar``, as the model holds them.
    That is the raw normal equations with the intercept unpenalized,
    exactly, where the target was present on every row the Gram holds, and
    it is what ``target_gaps="pairwise"`` solves where it was not. A
    mapping without ``cross_centered`` forms ``cross_moments[t] - m * ybar`` instead, which
    at a level ``L`` carries a rounding error of about ``L**2 * eps``, with
    ``eps`` the float64
    rounding unit. Without an intercept nothing is centered:
    ``(E[z z'] + ridge * I) b = E[z y]``, and every column is penalized.

    **A grid of ridges shares one eigendecomposition where the system's
    diagonal is even**, under ``standardize`` or on columns of like scale.
    With ``V d V'`` in hand, every ridge ``r`` is ``V diag(1/(d + r)) V' c``,
    so a grid of fifty takes one factorization and fifty matrix products.

    **The solve is accurate to the condition of the system with each column
    scaled by the root of its diagonal**, as the model's Cholesky is. So the
    features' units do not enter: columns from ``1e-10`` to ``1e10`` are
    solved as well as columns of unit scale. ``numpy.linalg.eigh`` is
    accurate only relative to the largest eigenvalue, so the system is
    scaled that way before it is decomposed, and the answer scaled back. A
    uniform ridge is not uniform once the columns are scaled. So where the
    diagonal of ``C + ridge * I`` spreads by more than a factor of 4,
    largest over smallest, each ridge takes a solve of its own: an LU solve
    of the scaled system for a ridge above 0, its eigendecomposition at 0.
    On columns from ``1e-3`` to ``1e3`` in scale, a grid of fifty ridges
    took 1.4 ms at 50 columns and 8.4 ms at 200, three times the single
    decomposition's time (measured on an Apple M4 Pro under load).
    """
    np = _np()
    ridges = np.atleast_1d(np.asarray(ridge, dtype=float))
    if not np.all(np.isfinite(ridges) & (ridges >= 0.0)):
        # A negative ridge solved a different, possibly indefinite, system
        # and returned it as the fit (review round 4, YB12).
        msg = (
            f"po.gram.solve: ridge must be finite and at least 0, but it is "
            f"{float(ridges[0]) if np.ndim(ridge) == 0 else ridges.tolist()!r}. Set "
            "ridge to 0 for no shrinkage, or to a finite positive value."
        )
        raise ValueError(msg)
    t = _target_index(g, target, "po.gram.solve")
    slots, icept = _feature_slots(g, features, "po.gram.solve")
    scalar = np.ndim(ridge) == 0
    k = len(_columns(g, "po.gram.solve"))

    means = np.asarray(g["means"], dtype=float)
    cross = np.asarray(g["cross_moments"], dtype=float)[t]
    como = _comoments(np, g)
    out = np.zeros((len(ridges), k))

    m, ybar = means, 0.0
    if icept >= 0:
        m = _means_of(np, g, t)
        ybar = cross[icept]
        a = como[np.ix_(slots, slots)]
        b = _cross_centered(np, g, t, m, ybar)[slots]
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
        cols = [slots[i] for i in keep]
        rows = _ridge_solves(np, a[kk] / np.outer(s[keep], s[keep]), b[keep] / s[keep], ridges)
        out[:, cols] = rows / s[keep]
    if icept >= 0:
        out[:, icept] = np.nan if _is_within(g) else ybar - out[:, slots] @ m[slots]
    return out[0] if scalar else out


#: How far apart a system's diagonal may be, largest over smallest, for a
#: ridge to ride the eigendecomposition of the system as given. That
#: decomposition's eigenvalues are accurate to ``u`` times the largest, which
#: in the coordinates the solve is accurate in -- each column scaled by the
#: root of its diagonal -- is a backward error up to this ratio times the
#: equilibrated solve's: four costs at most two bits.
_SHORTCUT_SPREAD = 4.0


def _ridge_solves(np: Any, a: Any, b: Any, ridges: Any) -> Any:
    """``(a + r I) x = b`` for each ridge ``r``, one row of ``x`` per ridge.

    A solve of a symmetric positive definite system is accurate in the
    coordinates that scale each column by the root of its diagonal: there
    the system is the equilibrated ``H = E^-1 a E^-1`` with a diagonal of 1,
    and its condition, not ``a``'s, is what the answer can lose. ``eigh``
    finds eigenvalues accurate to ``u`` times the largest, so on ``a`` as
    given, with columns from ``1e-10`` to ``1e10``, a small column's
    eigenvalue (``1e-20``) is lost beside ``1e20``. So the eigendecomposition
    is taken of ``H``, and scaled back: ``x = E^-1 V diag(1/d) V' E^-1 b``.

    A uniform ridge on ``a`` is a diagonal ``r E^-2`` on ``H``, which ``H``'s
    eigenvectors do not diagonalize, so one decomposition serves every ridge
    only on ``a`` as given. That shortcut is kept where it is accurate: where
    ``a + r I``'s diagonal spreads by no more than ``_SHORTCUT_SPREAD``,
    largest over smallest, so that equilibrating would change little. There
    ``x = V diag(1/(d + r)) V' b`` from ``a``'s own decomposition, taken once,
    as before. Elsewhere each ridge is solved on its own equilibrated ``a +
    r I``: by an LU solve where ``r > 0`` and the matrix is positive
    definite, by its eigendecomposition at ``r = 0``, where the system may be
    singular and the eigenvalues say so."""
    k = len(b)
    out = np.empty((len(ridges), k))
    diag = np.clip(np.diag(a), 0.0, None)

    def equilibrated(r: float) -> Any:
        e = np.sqrt(diag + r)
        e = np.where(e > 0.0, e, 1.0)
        h = (a + r * np.eye(k)) / np.outer(e, e)
        if r > 0.0:
            # Positive definite: an LU solve, backward stable on it and a
            # tenth of an eigendecomposition's work.
            try:
                return np.linalg.solve(h, b / e) / e
            except np.linalg.LinAlgError:
                pass
        d, v = np.linalg.eigh(h)
        return (v @ ((v.T @ (b / e)) / d)) / e

    raw = None
    for i, r in enumerate(ridges):
        spread = (diag.max() + r) / (diag.min() + r) if diag.min() + r > 0.0 else np.inf
        if spread <= _SHORTCUT_SPREAD:
            if raw is None:
                d, v = np.linalg.eigh(a)
                raw = (d, v, v.T @ b)
            d, v, vb = raw
            out[i] = v @ (vb / (d + r))
        else:
            out[i] = equilibrated(float(r))
    return out


@renamed_keywords("po.gram.lasso_path", {"lambdas": "penalties"})
def lasso_path(
    g: Gram,
    penalties: Sequence[float],
    *,
    l1_ratio: float = 1.0,
    penalty_weights: Sequence[float] | None = None,
    target: str | int = 0,
    features: Sequence[str | int] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> Any:
    """Compute the elastic-net path from a Gram, one row of coefficients per penalty.

    Use it to fit the lasso or the elastic net on a Gram over a grid of
    penalties you choose, as the ``lasso`` model does over its stream. For
    the start of the lasso path, knot by knot and exactly, use
    :func:`lars_path`. For a ridge, use :func:`solve`.

    .. rubric:: Parameters

    ``g``
        The Gram, as :meth:`~polars_online.ModelBank.gram` returns it.
    ``penalties``
        The penalties ``l``, from large to small, as a path is meant to be
        walked: each penalty's descent starts from the fit at the one before
        it, which makes the walk faster and better conditioned. They are a
        ``lasso`` spec's ``penalties``, and ``bank.coef()`` reports each as
        ``penalty``. Each is refused unless finite and at least 0. The
        keyword's older name, ``lambdas``, is refused, and the error names
        this one.
    ``l1_ratio``
        The share of the penalty that is L1. Default 1, the lasso; below 1,
        an elastic net. Refused outside ``[0, 1]``.
    ``penalty_weights``
        A scale on the penalty per feature, in ``features`` order (or in
        ``columns`` order without the intercept): 0 leaves a column
        unpenalized. A column with no variance in the Gram is dropped whatever
        its weight. Default ``None``, every weight 1. The ``lasso`` model
        takes no such parameter: here the path is walked again on every
        call, so each call may weigh the columns its own way. Refused at the
        wrong length, or with a weight that is not finite and at least 0.
    ``target``, ``features``
        As for :func:`solve`.
    ``max_iter``
        The most sweeps of the coordinate descent at each penalty. Default
        1,000, where the model's is 100, since this path starts its first
        penalty from zero and the model starts each solve from its last
        fit. Refused below 1.
    ``tol``
        The stopping rule: a sweep ends the descent when it moves no
        standardized coefficient by ``tol * s`` or more, or moves none at
        all, with ``s`` the target's spread (Notes). Default 1e-10, the
        model's. Refused unless finite and above 0.

    .. rubric:: Returns

    A float array of shape ``(len(penalties), k)`` over the Gram's ``k``
    columns, the intercept first when there is one, in the features'
    original units with the intercept recovered from the means, so a row is
    directly comparable to ``bank.coef()``. On a Gram pooled within parts
    (``merge(within=True)``) the intercept is ``nan``: such a Gram has an
    intercept per part and none of its own.

    .. rubric:: Raises

    ``ValueError``
        A number outside its range, by the rule a spec applies to the same
        parameter (each listed under Parameters); ``penalty_weights`` of the
        wrong length; the intercept or a column named twice among
        ``features``, as for :func:`solve`.
    ``KeyError``, ``IndexError``
        As for :func:`solve`.
    ``TypeError``
        ``penalties``' older name, ``lambdas``, is given.
    :class:`~polars_online.ReadinessWarning`
        A warning, not an error: issued for each penalty whose descent runs
        out of sweeps before it meets ``tol``. That row keeps the
        coefficients where the descent stopped. Increase ``max_iter``, or
        take the knots exactly from :func:`lars_path`.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("ridge")[0]
        path = po.gram.lasso_path(g, [1.0, 0.1, 0.001], target="y")   # one row per penalty
        print(path.round(3) + 0.0)                                  # + 0.0 prints -0.0 as 0.0

    .. code-block:: text

        [[ 0.966  0.784  0.   ]
         [ 0.991  1.848 -0.885]
         [ 0.992  1.968 -1.002]]

    .. rubric:: See also

    - :func:`lars_path`, the lasso path's knots, exactly.
    - :func:`solve`, a ridge on the same Gram.
    - *Choosing columns on the lasso path* in `docs/GRAM.md
      <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__,
      the theory, and `Choose columns on the lasso path
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/choose-columns-on-the-lasso-path.md#choose-columns-on-a-grams-lasso-path>`__,
      a recipe that runs.

    .. rubric:: Notes

    **The descent is the model's own**: the ``lasso`` model's, run on the standardized
    (correlation-form) matrix, each penalty starting from the fit at the
    one before it:

    .. code-block:: text

        b_i = soft(rho_i, l * l1_ratio * pw_i) / (C_ii + l * (1 - l1_ratio) * pw_i)
        soft(v, t) = sign(v) * max(|v| - t, 0)

    where ``b_i`` is column ``i``'s standardized coefficient, ``rho_i`` its
    standardized cross-correlation with the target less the other columns'
    contributions, ``C_ii`` its standardized variance and ``pw_i`` its
    penalty weight. With an intercept the system is centered at the
    target's own means, as the model's descent reads it. Through the origin
    nothing is centered and the raw moments are used, as the model does.

    **The stopping rule has no units.** ``s`` is the target's spread, read
    from the Gram's target moments: its standard deviation with an
    intercept, its root mean square through the origin. The standardized
    coefficients are in the target's units, so the target scaled by a power
    of two, with the penalties, gives the same path scaled, to the bit,
    stopped on the same sweep. ``s`` is never below the largest standardized
    cross-moment ``|d_i|``, which cannot exceed the target's spread, so a
    Gram without target moments has a scale too. On 15 features at seven
    penalties, ``tol=1e-10`` cost the same as 1e-7 (0.07 ms) and landed
    3e-10 from the converged path, where 1e-7 landed 4e-7.

    **A descent on columns that nearly repeat each other takes many
    sweeps.** A cyclic descent shrinks the error along two columns of
    correlation ``rho`` by ``rho**2`` a sweep. Their correlation matrix's
    condition number is ``kappa = (1 + rho) / (1 - rho)``, so ``rho**2 =
    ((kappa - 1) / (kappa + 1))**2``, about ``1 - 4 / kappa``, and each
    e-fold of the error takes about ``kappa / 4`` sweeps.

    **The descent runs in Rust.** On 40 penalties at 15 strongly correlated
    columns it took 1.8 ms, where the same descent as a Python loop took
    129 ms. Each sweep moves the coordinates in the loop's order, so a row
    matches the loop's to rounding (``2e-15`` there), not to the bit: the
    sum over the other columns runs in a different order.
    """
    np = _np()
    # Each was taken as given: `max_iter=0` returned the intercept with every
    # slope 0, a fully penalised fit to look at, and `tol=nan` never stopped
    # early (review round 4, YB12).
    lams = np.asarray(penalties, dtype=float).reshape(-1)
    if not np.all(np.isfinite(lams) & (lams >= 0.0)):
        msg = (
            "po.gram.lasso_path: penalties must be finite and at least 0, but they are "
            f"{lams.tolist()!r}. Set each penalty to 0 for no shrinkage, or to a finite "
            "positive value."
        )
        raise ValueError(msg)
    if not 0.0 <= l1_ratio <= 1.0:
        msg = (
            f"po.gram.lasso_path: l1_ratio must be from 0 to 1, but it is {l1_ratio!r}. Set "
            "l1_ratio from 0 (a ridge) to 1 (the lasso)."
        )
        raise ValueError(msg)
    if max_iter < 1:
        msg = (
            f"po.gram.lasso_path: max_iter must be at least 1, but it is {max_iter!r}, and "
            "with no sweep of the descent every coefficient but the intercept stays at 0. Set "
            "max_iter to 1 or more."
        )
        raise ValueError(msg)
    if not (math.isfinite(tol) and tol > 0.0):
        msg = (
            f"po.gram.lasso_path: tol must be finite and above 0, but it is {tol!r}. Set tol "
            "to a small positive value, such as the default, 1e-10."
        )
        raise ValueError(msg)
    t = _target_index(g, target, "po.gram.lasso_path")
    slots, icept = _feature_slots(g, features, "po.gram.lasso_path")
    k = len(_columns(g, "po.gram.lasso_path"))
    pw = (
        np.ones(len(slots)) if penalty_weights is None else np.asarray(penalty_weights, dtype=float)
    )
    if pw.shape != (len(slots),):
        msg = (
            f"po.gram.lasso_path: penalty_weights must have one entry for each of the "
            f"{len(slots)} features, but it has shape {pw.shape}. Pass one weight for each "
            "feature, in the order of features, or of the Gram's columns without the intercept "
            "when features is not set."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(pw) & (pw >= 0.0)):
        msg = (
            "po.gram.lasso_path: penalty_weights must be finite and at least 0, but they are "
            f"{pw.tolist()!r}. Set each weight to 0 to leave its feature unpenalized, or to a "
            "finite positive value."
        )
        raise ValueError(msg)
    flat, unconverged = _native.gram_cd_path(
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
    short = [float(v) for v, u in zip(lams, unconverged, strict=True) if u]
    if short:
        # Counted and said (task 278, D6): the rows were returned as the
        # descent left them with nothing to tell them apart.
        warnings.warn(
            ReadinessWarning(
                f"po.gram.lasso_path: the coordinate descent ran out of sweeps at {len(short)} "
                f"of {len(lams)} penalties, {short}, still moving a coefficient by tol times "
                f"the target's spread or more after max_iter = {max_iter} sweeps, so the path's "
                "rows at those penalties are where it stopped, short of the fit. Strongly "
                "correlated features slow the descent. Increase "
                "max_iter, or take the path's knots exactly from po.gram.lars_path."
            ),
            stacklevel=_user_stacklevel(),
        )
    return np.asarray(flat, dtype=float).reshape(len(lams), k)


def _native_gram(np: Any, g: Gram, icept: int) -> tuple[Any, ...]:
    """A Gram's arrays as the Rust fits read them: ``k`` and
    ``weight_sum``, then ``means``, ``comoments``, ``cross_moments``,
    ``means_by_target``, ``cross_centered``, ``target_weights``,
    ``target_means``, ``target_vars`` and ``target_n_kish``, each
    C-contiguous float64, and last ``target_within_parts`` as a list for a
    Gram pooled within parts, else ``None``. A mapping without ``means_by_target`` or
    ``cross_centered`` gets them as :func:`solve` forms them: ``means`` for
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
    cc = g.get("cross_centered")
    if cc is None or (len(cc) == 0 and m > 0):
        cc = np.zeros((m, k))
        if icept >= 0:
            for t in range(m):
                cc[t] = _cross_centered(np, g, t, by_target[t], cross[t][icept])
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
        [int(v) for v in g["target_within_parts"]] if _is_within(g) else None,
    )


def _limit(name: str, v: int | None, who: str) -> int | None:
    if v is None:
        return None
    # Any whole number, numpy's integers included (task 352), but not a bool.
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        msg = (
            f"{who}: {name} must be a whole number of at least 1, or None, but it is {v!r}. "
            "Pass a whole number of at least 1, or None for no limit."
        )
        raise WrongTypeError(msg)
    n = operator.index(v)
    if n < 1:
        msg = (
            f"{who}: {name} must be at least 1, or None, but it is {n}. Set {name} to 1 or "
            "more, or to None for no limit."
        )
        raise ValueError(msg)
    return n


def _lars_weights(
    np: Any, n: int, penalty_weights: Sequence[float] | None, who: str
) -> list[float]:
    if penalty_weights is None:
        return []
    pw = np.asarray(penalty_weights, dtype=float)
    if pw.shape != (n,):
        msg = (
            f"{who}: penalty_weights must have one entry for each of the {n} features, but it "
            f"has shape {pw.shape}. Pass one weight for each feature, in the order of features, "
            "or of the Gram's columns without the intercept when features is not set."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(pw) & (pw > 0.0)):
        msg = (
            f"{who}: penalty_weights must be finite and above 0, but they are "
            f"{pw.tolist()!r}, and a path cannot start from the largest penalty with a "
            "feature that no penalty removes. Set each weight above 0, or use "
            "po.gram.lasso_path for a weight of 0."
        )
        raise ValueError(msg)
    return [float(v) for v in pw]


@deprecated_keywords("po.gram.lars_path", {"max_steps": "max_iter"})
def lars_path(
    g: Gram,
    *,
    target: str | int = 0,
    features: Sequence[str | int] | None = None,
    max_iter: int | None = None,
    max_active: int | None = None,
    penalty_weights: Sequence[float] | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Compute the lasso path from its start, knot by knot, by least angle regression.

    Use it to find which columns the lasso chooses first, and the exact
    penalties at which each enters or leaves, without choosing a grid of
    penalties. For a grid of penalties, a penalty weight of 0 or an elastic
    net, use :func:`lasso_path`. For many Grams or targets in one call, use
    :func:`lars_paths`.

    .. rubric:: Parameters

    ``g``
        The Gram, as :meth:`~polars_online.ModelBank.gram` returns it.
    ``target``, ``features``
        As for :func:`solve`.
    ``max_iter``
        Stop after this many knots past the path's first, each knot one
        penalty however many columns change at it. Default ``None``, no
        limit. Refused unless a whole number of at least 1. The keyword's
        older name, ``max_steps``, is still read, with a
        :class:`~polars_online.PolarsOnlineDeprecationWarning`.
    ``max_active``
        Stop at the first knot with this many columns active or more (more
        than ``max_active`` where columns tie there). Default ``None``, no
        limit. Refused unless a whole number of at least 1. With neither
        limit, the path runs to ``l = 0``. A selection reads only the start
        of a path, and each step costs ``O(k |A|)`` operations (Notes), so
        stop it early.
    ``penalty_weights``
        A scale ``w_j`` on the penalty per feature, as for
        :func:`lasso_path`, but each finite and above 0. A weight of 0 would
        keep a column in the fit at every penalty, and a path that starts
        at ``l_max``, where every coefficient is 0, cannot hold such a
        column. Use :func:`lasso_path` for that, and for an elastic net, whose
        path is not piecewise linear. Default ``None``, every weight 1. Refused at the wrong length.
    ``threads``
        How many threads the path runs on, as :class:`~polars_online.ModelBank`
        takes it: ``None``, the default, is the process's threads, and an
        integer that many threads. One path is one thread's work, so it
        matters only to :func:`lars_paths`. The numbers are the same at any
        count. Refused unless an integer of at least 1, or ``None``.

    .. rubric:: Returns

    A dict:

    ``penalties``
        The penalty ``l`` at each knot, falling, in the units of
        :func:`lasso_path` and of the ``lasso`` model.
    ``coef``
        Shape ``(n_knots, k)``: each knot's coefficients over the Gram's
        ``k`` columns, in the features' original units with the intercept
        recovered, as a row of :func:`lasso_path` gives them (the intercept
        ``nan`` on a Gram pooled within parts).
    ``active``
        The active set below each knot, by name, in the order its columns
        entered, those entering at one knot in ``features`` order. A column
        entering at a knot is in that knot's set with its coefficient still
        0, and grows from there. A column leaving at a knot is out of its
        set. So the first knot's set holds the first columns to enter, one
        unless several tie, and a knot's nonzero coefficients are the set
        before it, less any column leaving at it. A tie that turns a slope
        back as it reaches 0 is the one case apart: that column stays in
        the set at 0.
    ``stop``
        Why the path stopped: ``"max_iter"``, ``"max_active"``, or
        ``"end"``, where ``l`` reached 0 or no column can enter.

    .. rubric:: Raises

    :class:`~polars_online.WrongTypeError`
        A limit that is not a whole number or ``None``. It is both a
        ``TypeError`` and a ``ValueError``.
    ``ValueError``
        A limit below 1; ``penalty_weights`` of the wrong length, or not
        finite and above 0; ``threads`` below 1; the intercept or a column
        named twice among ``features``, as for :func:`solve`.
    ``TypeError``
        ``threads`` that is not an integer or ``None``, such as a bool or a
        float; or ``max_iter`` and its older name ``max_steps`` given
        together.
    ``KeyError``, ``IndexError``
        As for :func:`solve`.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1", "x2"], half_life=100.0
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("ridge")[0]
        path = po.gram.lars_path(g, target="y")
        for l, active in zip(path["penalties"], path["active"]):
            print(f"{l:.3f}", active)       # a knot's penalty, and the columns active below it
        print(path["stop"])

    .. code-block:: text

        1.766 ['x0']
        0.848 ['x0', 'x1']
        0.016 ['x0', 'x1', 'x2']
        0.000 ['x0', 'x1', 'x2']
        end

    .. rubric:: See also

    - :func:`lars_paths`, the same for many Grams and targets at once.
    - :func:`lasso_path`, the lasso at penalties you choose.
    - *Choosing columns on the lasso path* in `docs/GRAM.md
      <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__,
      the theory, and `Choose columns on the lasso path
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/choose-columns-on-the-lasso-path.md#choose-columns-on-a-grams-lasso-path>`__,
      a recipe that runs.

    .. rubric:: Notes

    **The path is Efron, Hastie, Johnstone and Tibshirani's LARS (2004)
    with their lasso modification**, on the correlation form
    :func:`lasso_path` standardizes the Gram to. ``R`` is the features'
    correlation matrix and ``d`` their standardized cross-moments with the
    target, centered at the target's own means where there is an intercept.
    The lasso minimizes

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
    At ``l = 0`` the fit is least squares on ``A``.

    **A knot's coefficients are the lasso's exactly at its penalty**, so
    ``lasso_path(g, [l])`` at a knot's ``l`` gives the same row where its
    descent has converged. That descent's ``tol`` bounds a sweep's largest
    move, not the distance from the solution: at a condition number of 1e8
    a descent at ``tol=1e-14`` stopped 0.48 from a knot in the intercept, at
    an objective above the knot's.

    **Columns that tie change at one knot.** Where several columns reach
    the active correlation at once, as two with equal ``|d_j|`` do on an
    orthogonal design, they are decided together at that one knot. A tie is
    read to within rounding: steps within ``64 * eps * l_max`` of the
    smallest are one knot, since ``l`` is computed to about that. The lasso
    decides which tied columns enter. A tied column stays out when the
    direction that includes it would move its slope against its sign, and
    it can enter further down the path. scikit-learn 1.9.1's
    ``lars_path_gram`` takes tied columns one at a time and ends off least
    squares: on an identity Gram with ``d = [1, 1]``, at ``[2, 1]``, where
    least squares is ``[1, 1]``.

    **Some columns never enter.** A column that is constant on the Gram's
    rows never enters. A column that would enter collinear with the active
    set is set aside, since it adds no direction: its pivot in the factor
    of ``R_AA`` is below ``1e-12`` of its variance. It takes no step and
    makes no knot, so each knot but the last at ``l = 0`` is one column or
    more entering or leaving.

    **A path is fast.** A step takes ``O(k |A|)`` operations. Measured on
    an M4 Pro, the path's own arithmetic takes 9 microseconds whole at 15
    columns, and the call from Python 0.04 ms. At 200 columns, 40 knots take
    0.2 ms, and at 2,000 columns 3 ms.
    """
    return _lars_paths(
        [g],
        targets=[target],
        features=features,
        max_iter=max_iter,
        max_active=max_active,
        penalty_weights=penalty_weights,
        threads=threads,
        who="po.gram.lars_path",
    )[0][0]


@deprecated_keywords("po.gram.lars_paths", {"max_steps": "max_iter"})
def lars_paths(
    grams: Sequence[Gram],
    *,
    targets: Sequence[str | int] | None = None,
    features: Sequence[str | int] | None = None,
    max_iter: int | None = None,
    max_active: int | None = None,
    penalty_weights: Sequence[float] | None = None,
    threads: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Compute :func:`lars_path` for many Grams and targets at once, on the process's threads.

    Use it to walk the lasso path of every group's Gram, or of every target,
    in one call. To pool subsets of the Grams first and walk each pooled
    Gram's path, use :func:`solve_subsets`. For one path, :func:`lars_path`
    is the same arithmetic.

    .. rubric:: Parameters

    ``grams``
        The Grams, each as :meth:`~polars_online.ModelBank.gram` returns it,
        all with the same ``columns``. Each is fitted as it is: nothing is
        merged, and a Gram pooled within parts has its intercept ``nan``.
    ``targets``
        The targets to walk, by name or position, in each Gram. Default
        ``None``, every target of each Gram.
    ``features``, ``max_iter``, ``max_active``, ``penalty_weights``
        As for :func:`lars_path`, applied to every path, ``max_iter``'s
        older name ``max_steps`` included.
    ``threads``
        How many threads the Grams run on, beside each other: ``None``, the
        default, is the process's threads, which
        ``POLARS_ONLINE_MAX_THREADS`` sizes, and an integer that many
        threads, as :class:`~polars_online.ModelBank` takes it. Refused
        unless an integer of at least 1, or ``None``.

    .. rubric:: Returns

    One list per Gram, with one path per target, each the dict
    :func:`lars_path` returns. An empty ``grams`` gives an empty list.

    .. rubric:: Raises

    ``ValueError``
        Grams of different ``columns``, as for :func:`merge`.

    Otherwise, it raises what :func:`lars_path` raises.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                               half_life=100.0, group="stock_id")
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        paths = po.gram.lars_paths(bank.gram("ridge"), max_active=1)   # one list per stock
        # the first column to enter each stock's path
        print([per_gram[0]["active"][-1] for per_gram in paths])

    .. code-block:: text

        [['x0'], ['x0'], ['x0'], ['x0']]

    .. rubric:: See also

    - :func:`lars_path`, one path.
    - :func:`solve_subsets`, whose ``jobs`` give each path columns of its
      own, on pooled subsets of the Grams.

    .. rubric:: Notes

    Each Gram's column scaling is formed once, and its targets' paths run
    beside each other on it, each reading only its active columns' rows of
    the correlation matrix from the co-moments. A path is the same arithmetic on one
    thread whatever the number of threads, so its numbers are those of
    :func:`lars_path` to the bit. :func:`solve_subsets` with ``path`` and
    one Gram a subset, ``[[0], [1], ...]``, gives each subset the path
    :func:`lars_path` gives on that Gram, to the bit.
    """
    return _lars_paths(
        grams,
        targets=targets,
        features=features,
        max_iter=max_iter,
        max_active=max_active,
        penalty_weights=penalty_weights,
        threads=threads,
        who="po.gram.lars_paths",
    )


def _lars_paths(
    grams: Sequence[Gram],
    *,
    targets: Sequence[str | int] | None,
    features: Sequence[str | int] | None,
    max_iter: int | None,
    max_active: int | None,
    penalty_weights: Sequence[float] | None,
    threads: int | None,
    who: str,
) -> list[list[dict[str, Any]]]:
    """:func:`lars_paths`, its refusals naming ``who``, the call the reader
    made (:func:`lars_path` runs through here too)."""
    np = _np()
    threads = _check_threads(threads)
    parts = list(grams)
    if not parts:
        return []
    cols = _columns(parts[0], who)
    for i, p in enumerate(parts[1:], 1):
        if _columns(p, who) != cols:
            msg = (
                f"{who}: every Gram must have the same columns, but Gram 0 has {_listed(cols)} "
                f"and Gram {i} has {_listed(_columns(p, who))}. Pass Grams of specs with the "
                "same features."
            )
            raise ValueError(msg)
    slots, icept = _feature_slots(parts[0], features, who)
    steps, active = _limit("max_iter", max_iter, who), _limit("max_active", max_active, who)
    weights = _lars_weights(np, len(slots), penalty_weights, who)
    tidx = [
        list(range(len(p["cross_moments"])))
        if targets is None
        else [_target_index(p, t, who) for t in targets]
        for p in parts
    ]
    native = [_native_gram(np, p, icept) for p in parts]
    out = _native.gram_lars_paths(
        native, tidx, slots, None if icept < 0 else icept, weights, steps, active, threads
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


_PATH_KEYS = ("max_iter", "max_active", "penalty_weights")


def solve_subsets(
    grams: Sequence[Gram],
    subsets: Sequence[Sequence[int]],
    *,
    targets: Sequence[str | int] | None = None,
    features: Sequence[str | int] | None = None,
    ridge: float | Sequence[float] = 0.0,
    standardize: bool = False,
    path: dict[str, Any] | None = None,
    within: bool = False,
    target_signs: Any = None,
    threads: int | None = None,
    jobs: Sequence[Mapping[str, Any]] | pl.DataFrame | None = None,
) -> list[Any]:
    """Merge each subset of the Grams and fit it, for many subsets at once.

    Use it for many fits over pooled Grams in one call, such as each
    block's own fit (the spread of a coefficient's ``t`` across the blocks
    says how stable it is), the halves of a stability selection (each a
    merge of about half the blocks), and their null runs under random signs.
    For one merge and one fit, use :func:`merge` and then :func:`solve`,
    :func:`coef_stats` or :func:`lars_path`. To walk paths on Grams without
    merging them, use :func:`lars_paths`.

    .. rubric:: Parameters

    ``grams``
        The Grams, each as :meth:`~polars_online.ModelBank.gram` returns
        it, all with the same ``columns`` and ``targets``, and each with
        ``means_by_target`` and ``cross_centered``.
    ``subsets``
        One list of positions in ``grams`` per fit, each naming a Gram at
        most once, since a Gram twice in one merge counts its rows twice.
        ``[[0], [1], ...]`` fits each Gram alone. A subset's parts are
        pooled as :func:`merge` pools them.
    ``targets``
        The targets to fit, by name or position. Default ``None``, every
        target. Refused beside ``jobs``, whose jobs name their own.
    ``features``
        As for :func:`solve`.
    ``ridge``, ``standardize``
        As for :func:`solve`. A sequence of ridges fits each value, and
        gives ``coef``, ``se`` and ``t`` one row per value and
        ``resid_var``, ``sigma2`` and ``r2`` one entry per value, with ``n``
        once. Not read under ``path``, since a path is in correlation form.
    ``path``
        ``None``, the default, for a ridge fit with its statistics, as
        :func:`solve` and then :func:`coef_stats` give them. Or a dict of
        :func:`lars_path`'s ``max_iter``, ``max_active`` and
        ``penalty_weights`` (``{}`` for none of them) for each target's lasso
        path instead. The key's older name, ``max_steps``, is read as ``max_iter``,
        with a :class:`~polars_online.PolarsOnlineDeprecationWarning`. Any
        other key is refused.
    ``within``
        ``True`` pools each subset within its parts, as :func:`merge` with
        ``within=True``: each fit's intercept is ``nan``, and a ridge fit's
        degrees of freedom lose one per part. Default ``False``, the union.
    ``target_signs``
        ``None``, the default, or an array of shape ``(len(grams), draws)``,
        each entry +1 or -1, only with ``within=True``. Column ``d`` is a
        draw: a sign ``s_i`` for each Gram, which multiplies its targets'
        centered cross-moments before each subset is pooled,
        ``c = sum T_i s_i c_i / T``. That is what negating Gram ``i``'s
        target rows gives, since its co-moments, its target weights and its
        target variances do not move, so a draw needs no second merge of
        the co-moments. Every subset is fitted once per draw, so a draw of
        random signs is a null run of a selection: what it chooses where no
        column's effect keeps its sign across the Grams. Under a union a
        negated Gram's target mean moves the gaps between the parts' means
        too, and the shortcut does not hold, so it is refused there.
    ``threads``
        How many threads the subsets are fitted on, beside each other:
        ``None``, the default, is the process's threads, and an integer
        that many threads, as :class:`~polars_online.ModelBank` takes it.
        Each fit is one thread's arithmetic, so the numbers are the same at
        any count. Refused unless an integer of at least 1, or ``None``.
    ``jobs``
        ``None``, the default, fits every subset, draw and target on
        ``features``. Otherwise the fits to run, each on columns of its
        own: a list of mappings, as ``pl.DataFrame.to_dicts()`` gives them,
        or a Polars ``DataFrame`` of them, one row per job. A job's keys are

        - ``"subset"``, a position in ``subsets``;
        - ``"target"``, a name or position, as in ``targets``;
        - ``"draw"``, a column of ``target_signs``, given with it and only
          with it;
        - ``"features"``, the job's columns, as ``features`` names them;
          missing or ``None`` is ``features``, and so every feature when
          that is ``None`` too;
        - ``"penalty_weights"``, for a path only: one per column of the job.
          Missing or ``None`` is ``path``'s, which must then have one per
          column of the job too, or every weight 1.

        A job's fit is that of its columns merged alone, to the bit (Notes).

    .. rubric:: Returns

    One list per subset, with one dict per target. A ridge fit's dict has
    ``coef``, the coefficients over the Gram's columns, and
    :func:`coef_stats`' ``resid_var``, ``sigma2``, ``r2``, ``n``, ``se``
    and ``t``; under a sequence of ridges each number but ``n`` has one row
    or entry per ridge. A path's dict is :func:`lars_path`'s. With
    ``target_signs``, one such list per draw. With ``jobs``, one such dict per job, in their
    order, over every column of the Grams: 0, or ``nan`` for ``se`` and
    ``t``, at a column the job does not read.

    .. rubric:: Raises

    ``ValueError``
        When:

        - ``grams`` is empty, or the Grams differ in columns or targets;
        - a subset is empty, or holds a position out of range or twice;
        - without ``within``, a subset of several holds a Gram pooled within
          parts (as :func:`merge`);
        - ``target_signs`` is given without ``within``, has another shape, or
          holds an entry other than +1 or -1;
        - ``path`` has an unknown key, a limit below 1, or
          ``penalty_weights`` of the wrong length or not finite and above 0,
          as for :func:`lars_path`;
        - with ``jobs``: ``targets`` is given; a job has a key it does not
          take, lacks its ``subset`` or ``target``, has a ``subset`` or
          ``draw`` out of range, a ``draw`` without ``target_signs`` or none
          with them, ``features`` that are one name, hold the intercept or
          name a column twice, or ``penalty_weights`` without ``path``, of
          another length than its columns, or not finite and above 0;
        - a Gram has no ``means_by_target`` or ``cross_centered``, which
          :func:`merge` forms after pooling and this does not;
        - for a ridge fit, ``ridge`` is not finite and at least 0, or a Gram
          has no target moments (as :func:`coef_stats`);
        - ``threads`` is below 1.
    :class:`~polars_online.WrongTypeError`
        ``jobs`` is neither a list nor a ``DataFrame``, a job is not a
        mapping, or a limit in ``path`` is not a whole number. It is both a
        ``TypeError`` and a ``ValueError``.
    ``TypeError``
        ``threads`` is not an integer or ``None``, or ``path`` gives both
        ``max_iter`` and its older name ``max_steps``.
    ``KeyError``, ``IndexError``
        As for :func:`solve` and :func:`lars_path`: a job's column or target
        the Grams do not have is a ``KeyError``, and its position out of
        range an ``IndexError``, each naming the job.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                               half_life=float("inf"), group="stock_id")
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        blocks = bank.gram("ridge")                     # one Gram per stock
        # A ridge fit in each block alone, and x0's t in each.
        each = po.gram.solve_subsets(blocks, [[i] for i in range(len(blocks))], ridge=1e-6)
        print([round(float(fits[0]["t"][1]), 1) for fits in each])

    .. code-block:: text

        [38.8, 36.1, 37.8, 41.2]

    .. rubric:: See also

    - :func:`merge`, :func:`solve`, :func:`coef_stats` and :func:`lars_path`,
      one merge and one fit at a time.
    - *Stability selection over halves of the weeks*, *Per-block
      coefficients*, *A null for the selection, by sign flips* and *A
      screen of its own for each path* in `docs/GRAM.md
      <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__,
      the theory, and `Select columns that hold across the weeks
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/select-columns-stably-over-weeks.md>`__
      and `Fit per-block coefficients
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/fit-per-block-coefficients.md>`__,
      a recipe of each use.
    - *Pool within blocks to leave their levels out* in `Pool Grams
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/pool-grams.md>`__.

    .. rubric:: Notes

    **A path over each pair of blocks, its null runs under random signs,
    and a screen of its own for each job**, on the example's blocks:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                               half_life=float("inf"), group="stock_id")
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        blocks = bank.gram("ridge")                     # one Gram per stock
        halves = po.gram.solve_subsets(blocks, [[0, 1], [2, 3]], path={"max_active": 1})
        signs = np.random.default_rng(0).choice([-1.0, 1.0], size=(len(blocks), 20))
        null = po.gram.solve_subsets(blocks, [[0, 1], [2, 3]], path={"max_active": 1},
                                     within=True, target_signs=signs)   # a list per draw
        jobs = [{"subset": 0, "draw": 0, "target": "y", "features": ["x0"]},
                {"subset": 1, "draw": 3, "target": "y", "features": ["x1", "x0"]}]
        screened = po.gram.solve_subsets(blocks, [[0, 1], [2, 3]], path={"max_active": 1},
                                         within=True, target_signs=signs, jobs=jobs)
        print([h[0]["active"][-1] for h in halves],          # each half's first column
              len(null), len(null[0]),                       # 20 draws of 2 subsets
              [j["active"][-1] for j in screened])           # each job's first column

    .. code-block:: text

        [['x0'], ['x0']] 20 2 [['x0'], ['x0']]

    **The ridge fit** is

    .. code-block:: text

        (A + ridge * I) b = r_t            A, r_t as solve() forms them
        b_0       = ybar_t - m_t . b       the intercept, with one
        resid_var = S_yy - 2 b' r_t + b' A b
        sigma2    = resid_var * n / (n - p)      n = target_n_kish
        se        = sqrt(diag(inv(A)) * sigma2 / n),   t = b / se
        se_0      = sqrt((1 + m_t' inv(A) m_t) * sigma2 / n)

    with ``p`` the columns with variance and the intercept, or under
    ``within`` the columns with variance and the parts the target has rows
    in. A column of none, such as a constant that ``standardize`` drops at
    0, has no coefficient to estimate and spends no degree of freedom. With
    an intercept, ``A`` is the columns' centered co-moments, ``r_t`` the
    target's centered cross-moments, ``m_t`` its column means, ``ybar_t``
    its mean and ``S_yy`` its variance. Without one they are the raw
    moments and ``E[y**2]``, as :func:`coef_stats` reads them, and there is
    no ``se_0``. Under ``within`` the intercept has none either. Under
    ``standardize`` the system is solved in correlation form, as
    :func:`solve` solves it.

    **The ridge fit's system and the inverse its errors read are both
    factorized by Cholesky**, where :func:`solve` runs
    numpy's symmetric eigendecomposition and :func:`coef_stats` its
    inverse, so the two agree to rounding: the system's condition number
    times ``1e-16``, measured within ``5e-13`` of each number's size at
    2,000 columns. A system whose factorization fails, such as a constant
    column under a ridge of 0, gives ``nan`` here, where :func:`solve`
    divides by an eigenvalue of 0 or of rounding size. A nearly singular one
    gives what any solve of it gives, rounding noise. A column of no
    variance, such as a constant that ``standardize`` drops at 0, has
    ``nan`` for its ``se`` and ``t``, and the other columns' are read from
    the inverse of the columns kept, as a fit without it gives them.

    **The Grams are read once**, into memory the call holds while it runs,
    and every subset is merged and fitted there, on the process's threads,
    without a Python copy of a matrix per subset. The copy is the Grams' size again
    for the call's length. A subset's merge is formed on the columns its
    fits read and the intercept alone, and dropped when its fits are done.
    Its targets share one factorization. The merge is :func:`merge`'s
    arithmetic entry for entry, so it agrees with it to the bit. For 38
    Grams and 100 subsets of 19, measured on an M4 Pro on 2026-11-08 under
    a load near 10, it took 0.015 s at 200 columns and 3.3 s at 2,000, where
    the loop of :func:`merge`, :func:`solve` and :func:`coef_stats` took
    0.51 s and 80 s.

    **Jobs.** Each subset is merged once, on the intercept and every column
    its jobs read, and each draw pools its cross-moments again once per
    subset. A job's fit is that of its columns merged alone, to the bit: a
    path is :func:`lars_path` on ``subset(merge(parts, within=within),
    ["intercept", *features])`` (without ``"intercept"`` on a Gram through
    the origin) of the subset's Grams, each with its
    targets' cross-moments times its sign in the draw. A ridge fit is this
    call's on ``[subsets[subset]]``, ``target_signs[:, [draw]]``,
    ``features`` and ``targets=[target]``. Jobs that share a subset, draw
    and columns share that design's factorization, and the jobs run beside
    each other on the threads, as the subsets do. A screen per subset, per
    target and per draw, as complementary-pairs stability selection runs,
    is then one call. It was measured on an M4 Pro on 2026-11-08 under a
    load near 5, with 100 subsets of 19 of 38 Grams, 11 draws and 10
    targets: 11,000 jobs, each a path to 10 columns on 400 of 1,500
    columns. The call took 13 ms a subset. The loop took 700 ms a subset:
    :func:`merge` of the subset's signed Grams for each draw, then
    :func:`subset` and :func:`lars_path` for each job.
    """
    np = _np()
    threads = _check_threads(threads)
    parts = list(grams)
    who = "po.gram.solve_subsets"
    if not parts:
        msg = f"{who}: grams must hold at least one Gram, but it is empty. Pass the Grams to fit."
        raise ValueError(msg)
    cols = _columns(parts[0], who)
    names = list(parts[0].get("targets") or [])
    for i, p in enumerate(parts[1:], 1):
        if _columns(p, who) != cols:
            msg = (
                f"{who}: every Gram must have the same columns, but Gram 0 has {_listed(cols)} "
                f"and Gram {i} has {_listed(_columns(p, who))}. Pass Grams of specs with the "
                "same features."
            )
            raise ValueError(msg)
        if list(p.get("targets") or []) != names:
            msg = (
                f"{who}: every Gram must have the same targets, but Gram 0 has {names} and Gram "
                f"{i} has {list(p.get('targets') or [])}. Pass Grams of specs with the same "
                "targets."
            )
            raise ValueError(msg)
    picked: list[list[int]] = []
    for s in subsets:
        idx = [int(i) for i in s]
        if not idx:
            msg = (
                f"{who}: each subset must name at least one Gram, but one is empty. Give each "
                "subset the positions of the Grams to pool."
            )
            raise ValueError(msg)
        if any(not 0 <= i < len(parts) for i in idx):
            msg = (
                f"{who}: a subset's positions must be from 0 to {len(parts) - 1}, but subset "
                f"{list(s)!r} names one outside them. Name Grams by their position in grams."
            )
            raise ValueError(msg)
        if len(set(idx)) != len(idx):
            msg = (
                f"{who}: subset {list(s)!r} names a Gram twice, which would count its rows "
                "twice. Name each Gram once in a subset."
            )
            raise ValueError(msg)
        picked.append(idx)
        if not within and len(idx) > 1:
            _refuse_within_union(who, [parts[i] for i in idx])
    slots, icept = _feature_slots(parts[0], features, who)
    m = len(parts[0]["target_weights"])
    if jobs is not None and targets is not None:
        msg = (
            f"{who}: jobs and targets cannot both be set, since each job names its own target. "
            "Leave out targets, or jobs."
        )
        raise ValueError(msg)
    tidx = list(range(m)) if targets is None else [_target_index(parts[0], t, who) for t in targets]
    for gi, p in enumerate(parts):
        for key in ("means_by_target", "cross_centered"):
            v: Any = p.get(key)
            if m and (v is None or len(v) == 0):
                msg = (
                    f"{who}: every Gram must carry {key!r}, but Gram {gi} has none. Pass the Grams "
                    "as ModelBank.gram() returns them, or use po.gram.merge, then "
                    "po.gram.solve, which form it from cross_moments."
                )
                raise ValueError(msg)
    signs = _target_signs(np, target_signs, len(parts), within)
    if path is not None:
        # `lars_path`'s keyword renamed after 0.13, read under its old name
        # here too (task 244, M-24).
        path = forward_deprecated("po.gram.solve_subsets: path", path, {"max_steps": "max_iter"})
        unknown = sorted(set(path) - set(_PATH_KEYS))
        if unknown:
            msg = (
                f"{who}: path takes the keys {list(_PATH_KEYS)}, but it has {unknown}. Remove "
                "the other keys."
            )
            raise ValueError(msg)
    work = (
        None
        if jobs is None
        else _subset_jobs(
            np,
            parts[0],
            jobs,
            n_subsets=len(picked),
            n_draws=len(signs),
            default=slots,
            path=path,
        )
    )
    native = [_native_gram(np, p, icept) for p in parts]
    ic = None if icept < 0 else icept
    k = len(cols)

    def per_draw(out: list[Any]) -> list[Any]:
        # One list per draw from the native call; without signs, its one.
        return out if target_signs is not None else out[0]

    def path_dict(pen: list[float], coef: Any, act: list[list[int]], stop: str) -> dict[str, Any]:
        return {
            "penalties": np.asarray(pen, dtype=float),
            "coef": np.frombuffer(coef, dtype=np.float64).reshape(len(pen), k),
            "active": [[cols[i] for i in a] for a in act],
            "stop": stop,
        }

    if path is not None:
        steps = _limit("max_iter", path.get("max_iter"), f"{who}: path")
        active = _limit("max_active", path.get("max_active"), f"{who}: path")
        if work is not None:
            set_slots, set_weights, native_jobs = work
            done = _native.gram_path_jobs(
                native,
                picked,
                set_slots,
                set_weights,
                native_jobs,
                ic,
                steps,
                active,
                bool(within),
                signs,
                threads,
            )
            return [path_dict(*p) for p in done]
        weights = _lars_weights(np, len(slots), path.get("penalty_weights"), f"{who}: path")
        out = _native.gram_path_subsets(
            native, picked, tidx, slots, ic, weights, steps, active, bool(within), signs, threads
        )
        return per_draw([[[path_dict(*p) for p in per] for per in draw] for draw in out])
    scalar = np.ndim(ridge) == 0
    ridges = [float(r) for r in np.atleast_1d(np.asarray(ridge, dtype=float))]
    if not ridges or not all(math.isfinite(r) and r >= 0.0 for r in ridges):
        msg = (
            f"{who}: ridge must be finite and at least 0, but it is "
            f"{ridges[0] if scalar and ridges else ridges!r}. Set ridge to 0 "
            "for no shrinkage, or to a finite positive value."
        )
        raise ValueError(msg)
    for p in parts:
        if p.get("target_vars") is None or p.get("target_n_kish") is None:
            msg = (
                f"{who}: a ridge fit's standard errors read each target's variance, but a Gram "
                "has no target_vars or no target_n_kish. Pass the Grams as ModelBank.gram() "
                "returns them, which carry both."
            )
            raise ValueError(msg)

    def ridge_dict(
        coef: Any, se: Any, t: Any, resid_var: float, sigma2: float, r2: float, n: float
    ) -> dict[str, Any]:
        return {
            "coef": np.frombuffer(coef, dtype=np.float64),
            "resid_var": resid_var,
            "sigma2": sigma2,
            "r2": r2,
            "n": n,
            "se": np.frombuffer(se, dtype=np.float64),
            "t": np.frombuffer(t, dtype=np.float64),
        }

    if work is not None:
        set_slots, _, native_jobs = work

        def fit_jobs(r: float) -> list[dict[str, Any]]:
            done = _native.gram_ridge_jobs(
                native,
                picked,
                set_slots,
                native_jobs,
                ic,
                r,
                bool(standardize),
                bool(within),
                signs,
                threads,
            )
            return [ridge_dict(*f) for f in done]

        if scalar:
            return fit_jobs(ridges[0])
        # A grid, stacked per job as below per fit.
        job_grids = [fit_jobs(r) for r in ridges]
        return [
            {
                key: (
                    first[key]
                    if key == "n"
                    else np.stack([g[j][key] for g in job_grids])
                    if key in ("coef", "se", "t")
                    else np.asarray([g[j][key] for g in job_grids], dtype=float)
                )
                for key in first
            }
            for j, first in enumerate(job_grids[0])
        ]

    def fit(r: float) -> list[Any]:
        fits = _native.gram_ridge_subsets(
            native, picked, tidx, slots, ic, r, bool(standardize), bool(within), signs, threads
        )
        return [[[ridge_dict(*f) for f in per] for per in draw] for draw in fits]

    if scalar:
        return per_draw(fit(ridges[0]))
    # A grid, as :func:`solve` takes one: each value fitted in turn, and each
    # number stacked with one row per value; `n` does not depend on the ridge.
    grids = [fit(r) for r in ridges]
    return per_draw(
        [
            [
                [
                    {
                        key: (
                            first[key]
                            if key == "n"
                            else np.stack([g[d][s][t][key] for g in grids])
                            if key in ("coef", "se", "t")
                            else np.asarray([g[d][s][t][key] for g in grids], dtype=float)
                        )
                        for key in first
                    }
                    for t, first in enumerate(per)
                ]
                for s, per in enumerate(draw)
            ]
            for d, draw in enumerate(grids[0])
        ]
    )


def _target_signs(np: Any, target_signs: Any, n_grams: int, within: bool) -> list[list[float]]:
    """``solve_subsets``' ``target_signs`` checked and turned to one list of
    signs per draw, or ``[]`` for none."""
    if target_signs is None:
        return []
    if not within:
        msg = (
            "po.gram.solve_subsets: target_signs needs within=True, but within is False, and "
            "pooled as a union a Gram's sign would move the gaps between the Grams' means too. "
            "Pass within=True, or leave out target_signs."
        )
        raise ValueError(msg)
    s = np.asarray(target_signs, dtype=float)
    if s.ndim != 2:
        msg = (
            "po.gram.solve_subsets: target_signs must be a 2-d array, a row for each Gram and "
            f"a column for each draw, but it has shape {s.shape}. Pass one row for each Gram, "
            "such as signs[:, None] for one draw."
        )
        raise ValueError(msg)
    if s.shape[0] != n_grams:
        msg = (
            f"po.gram.solve_subsets: target_signs must have a row for each of the {n_grams} "
            f"Grams and a column for each draw, but it has shape {s.shape}. Pass one row for "
            "each Gram."
        )
        raise ValueError(msg)
    if s.shape[1] < 1:
        msg = (
            "po.gram.solve_subsets: target_signs must have at least one draw, a column of "
            "signs, but it has none. Pass at least one column."
        )
        raise ValueError(msg)
    if not np.all((s == 1.0) | (s == -1.0)):
        bad = s[~((s == 1.0) | (s == -1.0))][0]
        msg = (
            f"po.gram.solve_subsets: each entry of target_signs must be 1 or -1, but one is "
            f"{float(bad)!r}. Set every entry to 1 or -1."
        )
        raise ValueError(msg)
    return [[float(v) for v in col] for col in s.T]


_JOB_KEYS = ("subset", "target", "draw", "features", "penalty_weights")


def _position(v: Any, n: int) -> int | None:
    """``v`` as a position in ``0..n-1`` -- a whole number, numpy's
    included, but not a bool -- or ``None``."""
    if isinstance(v, bool):
        return None
    try:
        i = operator.index(v)
    except TypeError:
        return None
    return i if 0 <= i < n else None


def _subset_jobs(
    np: Any,
    g: Gram,
    jobs: Any,
    *,
    n_subsets: int,
    n_draws: int,
    default: list[int],
    path: dict[str, Any] | None,
) -> tuple[list[list[int]], list[list[float]], list[tuple[int, int, int, int]]]:
    """``solve_subsets``' ``jobs`` checked and turned to what the native
    call reads: the distinct column sets (their slots, and a path's penalty
    weights, ``[]`` for every weight 1), and per job its subset, target,
    draw and set, each a position. ``default`` is the slots of the call's
    ``features``, a job's columns when it names none."""
    if not isinstance(jobs, Sequence):
        import polars as pl

        if not isinstance(jobs, pl.DataFrame):
            msg = (
                "po.gram.solve_subsets: jobs must be a list of mappings, one for each fit, or "
                f"a frame of them, but its type is {type(jobs).__name__}. Pass a list of dicts or "
                "a frame."
            )
            raise WrongTypeError(msg)
        jobs = jobs.to_dicts()
    names = _columns(g, "po.gram.solve_subsets")
    icept = names.index(INTERCEPT) if INTERCEPT in names else -1
    where: dict[str, int] = {}
    for i, name in enumerate(names):
        where.setdefault(name, i)
    path_weights = None if path is None else path.get("penalty_weights")
    sets: dict[tuple[tuple[int, ...], tuple[float, ...]], int] = {}
    set_slots: list[list[int]] = []
    set_weights: list[list[float]] = []
    out: list[tuple[int, int, int, int]] = []
    for n, job in enumerate(jobs):
        who, job_at = "po.gram.solve_subsets", f"jobs[{n}]"
        at = f"{who}: {job_at}"
        if not isinstance(job, Mapping):
            msg = (
                f"{who}: {job_at} must be a mapping of {list(_JOB_KEYS)}, but its type "
                f"is {type(job).__name__}. Pass each job as a dict."
            )
            raise WrongTypeError(msg)
        unknown = sorted(str(key) for key in job if key not in _JOB_KEYS)
        if unknown:
            msg = (
                f"{who}: {job_at} has the keys {unknown}, but a job takes only {list(_JOB_KEYS)}. "
                "Remove the other keys."
            )
            raise ValueError(msg)
        for needed in ("subset", "target"):
            if job.get(needed) is None:
                msg = (
                    f"{who}: {job_at} names no {needed!r}, but every job names its subset and its "
                    f"target. Give the job a {needed!r}."
                )
                raise ValueError(msg)
        subset = _position(job["subset"], n_subsets)
        if subset is None:
            msg = (
                f"{who}: the 'subset' of {job_at} must be a position in subsets, from 0 to "
                f"{n_subsets - 1}, but it is {job['subset']!r}. Set it to the position of a "
                "subset."
            )
            raise ValueError(msg)
        target = _target_index(g, job["target"], at)
        draw = job.get("draw")
        if n_draws == 0:
            if draw is not None:
                msg = (
                    f"{who}: {job_at} names draw {draw!r}, but target_signs is not set, and a "
                    "draw is a column "
                    "of target_signs. Leave out 'draw', or pass target_signs."
                )
                raise ValueError(msg)
            draw = 0
        elif draw is None:
            msg = (
                f"{who}: {job_at} names no 'draw', but with target_signs every job names its "
                "column of signs. "
                "Give the job a 'draw'."
            )
            raise ValueError(msg)
        else:
            column = _position(draw, n_draws)
            if column is None:
                msg = (
                    f"{who}: the 'draw' of {job_at} must be a column of target_signs, from 0 to "
                    f"{n_draws - 1}, but it is {draw!r}. Set it to the position of a column."
                )
                raise ValueError(msg)
            draw = column
        feats = job.get("features")
        if feats is None:
            slots = default
        elif isinstance(feats, str):
            msg = (
                f"{who}: the 'features' of {job_at} must be a list of columns, but it is the one "
                f"name {feats!r}. Pass it in a list, as [{feats!r}]."
            )
            raise ValueError(msg)
        else:
            try:
                slots = [where[c] for c in feats]
            except (KeyError, TypeError):
                slots = _col_index(g, feats, at)
            if icept in slots:
                msg = (
                    f"{who}: the 'features' of {job_at} name {INTERCEPT!r}, but it is the "
                    "intercept's constant column, "
                    "which the fit adds itself. Leave it out of 'features'."
                )
                raise ValueError(msg)
            twice = _repeated(names, slots)
            if twice is not None:
                msg = (
                    f"{who}: the 'features' of {job_at} name column {twice!r} twice, and a fit "
                    "given one column "
                    "twice has no single answer, so one coefficient would overwrite the other. "
                    "Name each column once."
                )
                raise ValueError(msg)
        own = job.get("penalty_weights")
        weights: list[float] = []
        if path is None:
            if own is not None:
                msg = (
                    f"{who}: {job_at} has 'penalty_weights', but only a path reads them, and path "
                    "is not set. Pass path={} for the lasso path, or leave out 'penalty_weights'."
                )
                raise ValueError(msg)
        else:
            if own is None and feats is not None and path_weights is not None:
                try:
                    shape = np.shape(path_weights)
                except ValueError:
                    shape = None
                if shape is not None and shape != (len(slots),):
                    entries = f"{shape[0]} entries" if len(shape) == 1 else f"shape {shape}"
                    msg = (
                        f"{who}: {job_at} names {len(slots)} columns in its 'features', but "
                        f"path's penalty_weights has {entries}. Give {job_at} its own "
                        f"'penalty_weights' of {len(slots)} entries, or leave out its 'features'."
                    )
                    raise ValueError(msg)
            weights = _lars_weights(np, len(slots), path_weights if own is None else own, at)
        columns = (tuple(slots), tuple(weights))
        set_id = sets.get(columns)
        if set_id is None:
            set_id = sets[columns] = len(set_slots)
            set_slots.append(list(slots))
            set_weights.append(weights)
        out.append((subset, target, draw, set_id))
    return set_slots, set_weights, out


def coef_stats(
    g: Gram,
    coef: Sequence[float],
    *,
    target: str | int = 0,
    features: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    """Compute the residual variance, standard errors and t-statistics of a fit from a Gram.

    Use it to judge a fit read off a Gram, such as one :func:`solve`
    returns: how good it is, and which coefficients are real, without the
    rows. For many merged subsets of Grams at once, :func:`solve_subsets`
    returns the same statistics beside each fit.

    .. rubric:: Parameters

    ``g``
        The Gram, as :meth:`~polars_online.ModelBank.gram` returns it, with
        its target moments (``target_vars`` and ``target_n_kish``).
    ``coef``
        The fit, one entry per Gram column, as :func:`solve` returns it.
    ``target``
        The target the fit predicts, by name or position. Default 0, the
        first.
    ``features``
        The columns fitted, as for :func:`solve`. The other columns' entries
        in ``coef`` are not read, and their ``se`` and ``t`` are ``nan``.
        Default ``None``, every column.

    .. rubric:: Returns

    A dict:

    ``resid_var``, ``sigma2``, ``r2``
        The residual variance of the fit, its estimate corrected for the
        degrees of freedom, and ``1 - resid_var / Var[y]``.
    ``n``
        Kish's effective sample size, which the correction and the errors
        use (``target_n_kish``).
    ``se``, ``t``
        Arrays over the same columns as ``coef``, the intercept's included
        (``nan`` on a Gram pooled within parts), and ``nan`` for a column
        not fitted. A fitted column of no variance, such as a constant that
        :func:`solve`'s ``standardize`` drops at 0, has ``nan`` too, and the others' are
        read from the inverse of the columns with variance, as a fit without
        it gives them. A ``coef`` that is ``nan``, such as :func:`solve`'s
        on a constant column without ``standardize``, gives ``nan``
        throughout.

    .. rubric:: Raises

    ``ValueError``
        A ``coef`` without one entry per Gram column; a Gram without target
        moments, such as a mapping built by hand without them
        (:meth:`~polars_online.ModelBank.gram` reports both); the intercept
        or a column named twice among ``features``, as for :func:`solve`.
    ``KeyError``, ``IndexError``
        As for :func:`solve`.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("ridge")[0]
        coef = po.gram.solve(g, target="y")                 # least squares
        stats = po.gram.coef_stats(g, coef, target="y")
        print(stats["se"].round(3), stats["t"].round(1), round(stats["r2"], 3))

    .. code-block:: text

        [0.031 0.033 0.032] [ 31.5  60.6 -31.7] 0.943

    .. rubric:: See also

    - :func:`solve`, the fit.
    - :func:`solve_subsets`, the fit and these statistics for many merged
      subsets.
    - *A ridge and its t* in `docs/GRAM.md
      <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__,
      the theory, and `Fit a ridge from a Gram
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/fit-a-ridge-from-a-gram.md>`__,
      a recipe that runs.

    .. rubric:: Notes

    **The statistics** read the target's variance ``Var[y]`` from the Gram:

    .. code-block:: text

        resid_var = Var[y] - 2 b' Cov[X, y] + b' C b
        sigma2    = resid_var * n / (n - k)          # n = target_n_kish
        se        = sqrt(diag(inv(C)) * sigma2 / n)
        se_0      = sqrt((1 + m' inv(C) m) * sigma2 / n)   # the intercept's
        t         = b / se
        r2        = 1 - resid_var / Var[y]

    ``b`` is the slopes, ``C`` the features' centered co-moments,
    ``Cov[X, y]`` the target's centered cross-moments (``cross_centered``)
    and ``m`` the column means over the target's rows. ``k`` counts the
    features with variance and the intercept: a column of none, such as a
    constant that ``standardize`` drops at 0, has no coefficient to
    estimate and spends no degree of freedom. ``se_0`` is least squares'
    error of the intercept, ``ybar - m . b``: the mean and the slopes are
    uncorrelated, so its variance is ``Var[ybar] + m' Var[b] m``. On a Gram
    pooled within parts (``merge(within=True)``), ``k`` counts the features
    with variance and one intercept per part the target has rows in
    (``target_within_parts``). The intercept then has no error, since each
    part had its own. The fit's ``r2`` is the share of the variance inside
    the parts that it explains.

    **For a ridge fit, the standard errors are those of least squares, at
    the ridge fit's residual variance.** ``inv(C)`` is the inverse of the
    unpenalized matrix, so the errors take no account of the shrinkage,
    which makes a ridge's coefficients less variable than least squares',
    nor of the bias it adds. At a ridge near 0 they are least squares' own.

    **Without an intercept, the statistics read the raw moments, as least
    squares through the origin does.** On a Gram without an intercept
    column, as a :func:`subset` without ``"intercept"`` gives, ``C`` is
    ``E[z z']``, the co-moments plus the outer product of the means,
    ``Cov[X, y]`` is ``E[z y]`` (``cross_moments``), and ``Var[y]`` is
    ``E[y**2]``. So ``resid_var`` is the residuals' mean square, ``se`` and
    ``t`` are least squares' through the origin, and ``r2`` is uncentered,
    ``1 - E[e**2] / E[y**2]``, as statsmodels reports a fit with no
    constant. ``k`` counts the features alone, those not identically 0. On
    a Gram pooled within parts the means are 0, the target's too, so the
    raw moments are the centered ones and nothing moves.

    **The sample size is Kish's effective sample size, not the weight.**
    ``n`` is ``target_n_kish``, not ``weight_sum``: a weighted stream's
    weight sum is not a count, and dividing by it would report standard
    errors too small by the factor the weights are unequal by. The rows
    behind an exponentially weighted fit are also neither independent nor
    identically distributed, so read a ``t`` here as a scale for comparing
    coefficients, not as a p-value.
    """
    np = _np()
    t = _target_index(g, target, "po.gram.coef_stats")
    slots, icept = _feature_slots(g, features, "po.gram.coef_stats")
    if g.get("target_vars") is None or g.get("target_n_kish") is None:
        msg = (
            "po.gram.coef_stats: the standard errors read the target's variance, but the "
            "Gram has no target_vars or no target_n_kish. Pass the Gram as ModelBank.gram() "
            "returns it, which carries both."
        )
        raise ValueError(msg)

    beta = np.asarray(coef, dtype=float)
    k = len(_columns(g, "po.gram.coef_stats"))
    if beta.shape != (k,):
        msg = (
            f"po.gram.coef_stats: coef must have one entry for each of the Gram's {k} columns, "
            f"but it has shape {beta.shape}. Pass the coefficients po.gram.solve returns."
        )
        raise ValueError(msg)
    b = beta[slots]
    cross = np.asarray(g["cross_moments"], dtype=float)[t]
    como = _comoments(np, g)
    c = como[np.ix_(slots, slots)]
    var_y = float(np.asarray(g["target_vars"], dtype=float)[t])
    n = float(np.asarray(g["target_n_kish"], dtype=float)[t])
    m = _means_of(np, g, t)
    if icept >= 0:
        # Centred cross-covariance over the target's rows, the pair
        # `comoments` is in, as the model holds it (N4).
        a = c
        rhs = _cross_centered(np, g, t, m, cross[icept])[slots]
        s_yy = var_y
    else:
        # Through the origin, the raw moments solve() reads: the residuals'
        # mean square, not their variance, and an uncentred R^2.
        if g.get("target_means") is None:
            msg = (
                "po.gram.coef_stats: a fit with no intercept reads the target's mean as well "
                "as its variance, but the Gram has no target_means. Pass the Gram as "
                "ModelBank.gram() returns it, or keep the intercept column among its columns."
            )
            raise ValueError(msg)
        means = np.asarray(g["means"], dtype=float)[slots]
        a = c + np.outer(means, means)
        rhs = cross[slots]
        ybar = float(np.asarray(g["target_means"], dtype=float)[t])
        s_yy = var_y + ybar * ybar

    resid_var = s_yy - 2.0 * b @ rhs + b @ a @ b
    resid_var = max(resid_var, 0.0)
    # The columns with variance: one of none, such as a constant that
    # standardize dropped, has a coefficient of 0 by construction, spends no
    # degree of freedom, and is a zero row no inverse of the whole takes, so
    # it keeps its nan (task 270, review round 9 I-2).
    kept = np.flatnonzero(np.diag(a) > 0.0)
    # A Gram pooled within parts spent an intercept on each part, and has
    # none of its own (docs/PLAN.md task 240).
    if _is_within(g):
        absorbed = int(np.asarray(g["target_within_parts"])[t])
    else:
        absorbed = 1 if icept >= 0 else 0
    dof = n - (len(kept) + absorbed)
    sigma2 = resid_var * n / dof if dof > 0.0 else np.nan
    se = np.full(k, np.nan)
    tstat = np.full(k, np.nan)
    if np.isfinite(sigma2) and n > 0.0:
        a_kept = a[np.ix_(kept, kept)]
        try:
            inv = np.linalg.inv(a_kept)
        except np.linalg.LinAlgError:
            inv = np.full_like(a_kept, np.nan)
        se[np.asarray(slots, dtype=int)[kept]] = np.sqrt(
            np.clip(np.diag(inv), 0.0, None) * sigma2 / n
        )
        if icept >= 0 and not _is_within(g):
            # Least squares' inverse moment matrix at the intercept: ybar and
            # the slopes are uncorrelated, so Var[b_0] = Var[ybar] + m' Var[b] m.
            mt = m[slots][kept]
            se[icept] = np.sqrt((1.0 + mt @ inv @ mt) * sigma2 / n)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(se > 0.0, beta / se, np.nan)
    return {
        "resid_var": resid_var,
        "sigma2": sigma2,
        "r2": 1.0 - resid_var / s_yy if s_yy > 0.0 else float("nan"),
        "n": n,
        "se": se,
        "t": tstat,
    }


def vif(g: Gram, *, features: Sequence[str | int] | None = None) -> Any:
    """Compute each column's variance inflation factor from a Gram.

    Use it to find the columns whose coefficients a fit cannot pin down,
    because the other columns explain them. To see which columns share a
    near-dependency, use :func:`condition`. To read the correlations
    themselves, use :func:`correlation`.

    .. rubric:: Parameters

    ``g``
        The Gram, as :meth:`~polars_online.ModelBank.gram` returns it.
    ``features``
        The columns to diagnose, by name or position. Default ``None``, the
        Gram's columns without the intercept: a constant is explained
        perfectly by any other constant, so its factor is undefined. The
        intercept, or a column named twice, is refused.

    .. rubric:: Returns

    A float array over ``features``: ``1 / (1 - R2_j)`` for each column
    ``j``, with ``R2_j`` the share of its variance the other columns
    explain. A column the model found constant reports ``inf``. Above
    about 10, a common rule of thumb, the column's coefficient has more than
    ten times the variance it would have on uncorrelated columns. Fit such a
    column with a ridge, on a subset, or as a feature set the spec fits
    beside the full one (``ewridge``'s ``feature_sets``). Read anything past
    1e10 as an exact dependency (Notes).

    .. rubric:: Raises

    ``ValueError``
        The intercept or a column named twice among ``features``.
    ``KeyError``
        A name the Gram does not have.
    ``IndexError``
        A position outside the Gram's columns.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1", "x2"], half_life=100.0
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df.with_columns(x2=pl.col("x0") + 0.01 * pl.col("x2")))  # x2 nearly x0
        print([f"{v:.0f}" for v in po.gram.vif(bank.gram("ridge")[0])])   # x0, x1, x2

    .. code-block:: text

        ['10432', '1', '10432']

    .. rubric:: See also

    - :func:`condition`, which says which columns share a near-dependency.
    - *Is the fit determined?* in `docs/GRAM.md
      <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__,
      the theory, and `Check that a fit is determined
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/check-a-fit-is-determined.md>`__,
      a recipe that runs.

    .. rubric:: Notes

    **The factors are the diagonal of the inverse correlation matrix**,
    read from its eigendecomposition ``R = V diag(d) V'`` as
    ``VIF_j = sum_i V_ji^2 / d_i``. An eigenvalue at or below numpy's rank
    tolerance, ``d_max * k * eps``, with ``k`` the number of columns and
    ``eps`` the float64 rounding unit, is a dependency: a column with a
    weight past ``sqrt(eps)`` in its direction reports ``inf``. A
    pseudo-inverse would drop that direction instead, and it read an exact
    duplicate at 0.25, below the floor of 1 a factor cannot go under.

    **A column that is an exact linear combination of the others reads**
    ``inf`` **or a number past 1e10, not always** ``inf``. Its ``R2_j`` is
    1, but rounding in the accumulated moments can leave the dependency's
    eigenvalue just above the tolerance, and the column then reads about
    1e14. On 5,000 rows, a copy times 100 plus 3 read 2.2e14 at ``lam =
    0.999`` and ``inf`` at ``lam = 1``. Three columns where the third is the
    sum of the first two read 1.2e14 to 2.4e14 at ``lam = 1`` and ``inf`` at
    0.999.

    **A factor near 1 does not say a column is in no dependency.** A column
    whose part in a dependency is tiny reads near 1, as if it stood apart.
    Take ``c = a + b`` with ``a`` spread a million times as wide as ``b``:
    ``b``'s weight in the dependency's direction, in correlation units, is
    5e-13, far below ``sqrt(eps)``. It read 1.0002 where ``a`` and ``c``
    read ``inf``.
    """
    np = _np()
    slots, _ = _feature_slots(g, features, "po.gram.vif")
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


def condition(g: Gram, *, features: Sequence[str | int] | None = None) -> dict[str, Any]:
    """Compute Belsley's collinearity diagnostics for the design a Gram holds.

    Use it to find which columns share a near-dependency, and how close to
    singular the design is. :func:`vif` says only that a column is in a
    dependency, and a single condition number only that there is one.

    .. rubric:: Parameters

    ``g``
        The Gram, as :meth:`~polars_online.ModelBank.gram` returns it.
    ``features``
        The columns to diagnose, by name or position. Default ``None``,
        every column of the Gram, the intercept included.

    .. rubric:: Returns

    A dict:

    ``columns``
        The columns diagnosed, in order.
    ``singular_values``
        The singular values of the column-scaled design, largest first.
    ``condition_indexes``
        ``s_max / s_j`` for each singular value ``s_j``.
    ``kappa``
        The largest condition index.
    ``proportions``
        The variance-decomposition proportions: one row per component and
        one column per column diagnosed, each column summing to 1.

    A component with a large condition index and a large share of two or
    more columns' variance is a near-dependency between exactly those
    columns. Belsley's rule of thumb is an index above 30 with two
    proportions above 0.5.

    .. rubric:: Raises

    ``KeyError``
        A name the Gram does not have.
    ``IndexError``
        A position outside the Gram's columns.
    ``ValueError``
        A column named twice, by name or by position: two copies of one
        column are an exact dependency, which needs no diagnosis.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1", "x2"], half_life=100.0
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df.with_columns(x2=pl.col("x0") + 0.01 * pl.col("x2")))  # x2 nearly x0
        diag = po.gram.condition(bank.gram("ridge")[0])
        print(round(diag["kappa"]), diag["proportions"][-1].round(2))  # the last component's

    .. code-block:: text

        208 [0. 1. 0. 1.]

    .. rubric:: See also

    - :func:`vif`, each column's variance inflation.
    - *Is the fit determined?* in `docs/GRAM.md
      <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md>`__,
      the theory, and `Check that a fit is determined
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/check-a-fit-is-determined.md>`__,
      a recipe that runs.

    .. rubric:: Notes

    The design is scaled to unit column length first, Belsley's
    prescription, but not centered: the intercept is part of the
    collinearity when a column is nearly constant, and centering hides that.
    ``singular_values`` are of that scaled raw matrix, so they are the
    square roots of the eigenvalues of the scaled second-moment matrix.
    """
    np = _np()
    names = _columns(g, "po.gram.condition")
    slots = (
        list(range(len(names)))
        if features is None
        else _col_index(g, features, "po.gram.condition")
    )
    _refuse_repeated(names, slots, "po.gram.condition", "features")
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
