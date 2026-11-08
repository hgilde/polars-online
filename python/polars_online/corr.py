"""Read, repair, summarize and score correlation matrices.

:mod:`polars_online.gram` solves and diagnoses the design a Gram holds.
This module takes a correlation matrix, from a Gram or from anywhere else,
and does the arithmetic that comes after a pass:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - what you want
     - the calls
   * - read one
     - :func:`polars_online.gram.correlation`, from a Gram, and
       :func:`polars_online.gram.from_row` for a closed group's row
   * - repair it
     - :func:`nearest`, the nearest correlation matrix; :func:`shrink`,
       toward a structured target
   * - summarize it
     - :func:`equicorr`, :func:`equicorr_row` and :func:`equicorr_loglik`
       for one common correlation; :func:`block_means` and
       :func:`from_blocks` for blocks; :func:`spectral` and
       :func:`from_spectral` for the top eigenpairs; :func:`absorption`
       and :func:`absorption_shift` for the share the top eigenvectors take
   * - tell it from noise
     - :func:`mp_edge` and :func:`mp_density`, the Marchenko-Pastur
       edges and curve; :func:`signal_share` across blocks;
       :func:`fisher_se` for one correlation, with :func:`to_z` and
       :func:`from_z`
   * - score a forecast of it
     - :func:`loss`
   * - undo the Epps attenuation
     - :func:`epps_invert`

Every function is a pure function of its inputs: arrays or numbers, or a
Gram or a closed group's row. Where it takes a correlation matrix, it also
takes one of the Grams :meth:`~polars_online.ModelBank.gram` returns, or a row of
:meth:`~polars_online.ModelBank.closed_groups`, and reads its correlation
matrix as :func:`polars_online.gram.correlation` does. That matrix gives a
Gram's ``intercept`` column a row and a column of ``nan``, which each
function meets in its own way:

- :func:`nearest` drops them;
- :func:`equicorr` leaves them out of its mean, and :func:`block_means` out
  of each block's, which is ``nan`` where every entry behind it is;
- :func:`absorption` and :func:`spectral` raise numpy's ``LinAlgError``;
- :func:`shrink` returns ``nan`` in the intercept's row and column (the
  whole matrix toward ``target="identity"``), and its estimated intensity
  comes back as 0, which shrinks nothing;
- :func:`loss` gives ``nan``.

So pass such a Gram through :func:`polars_online.gram.subset` first, naming
its features.

The arithmetic is the papers', named in each docstring, and every function
is held against a longhand check in ``tests/test_corr.py``: Higham's own
published examples for :func:`nearest`, Ledoit and Wolf's formulas for
:func:`shrink`, and the closed forms for the rest. Where a library computes
the same thing, ``tests/test_second_opinion.py`` holds the function to it
too: statsmodels' ``corr_nearest`` for :func:`nearest`, and scikit-learn's
``ledoit_wolf`` for :func:`shrink` toward the identity.

The module needs numpy, an optional extra of this package (``pip install
polars-online[numpy]``) and not a dependency, since it is not one of
Polars' either. Nothing here needs scipy or scikit-learn.

.. warning::
    This module is considered **unstable**: its functions' names, keywords
    and results may change in any release without that counting as a
    breaking change. Under ``POLARS_ONLINE_WARN_UNSTABLE=1`` each call issues
    :class:`polars_online.UnstableWarning`, once per call, even when that
    call runs other functions of this module.
"""

from __future__ import annotations

import numbers
import operator
from collections.abc import Sequence
from typing import Any

from polars_online._errors import WrongTypeError
from polars_online._renamed import renamed_function
from polars_online._warnings import deprecated_alias, unstable

__all__ = [
    "Z_CLIP",
    "absorption",
    "absorption_shift",
    "block_means",
    "epps_invert",
    "equicorr",
    "equicorr_loglik",
    "equicorr_row",
    "fisher_se",
    "from_blocks",
    "from_spectral",
    "from_z",
    "loss",
    "mp_density",
    "mp_edge",
    "nearest",
    "shrink",
    "signal_share",
    "spectral",
    "to_z",
]

#: Fisher's transform is infinite at ±1, and a degenerate block gives
#: exactly that. Inputs are clipped here, so ``to_z(1.0)`` is about 7.25
#: rather than ``inf``. The round trip through :func:`from_z` is then not
#: quite the identity at the ends: it gives back ``±Z_CLIP``, to rounding,
#: for ``±1``.
Z_CLIP = 1.0 - 1e-6


def _np() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as e:  # pragma: no cover - exercised by a stub
        msg = (
            "po.corr: it works in numpy arrays, but numpy is not installed. "
            "Install it with `pip install numpy` or `pip install polars-online[numpy]`."
        )
        raise ModuleNotFoundError(msg) from e
    return np


def _matrix(obj: Any, who: str = "po.corr", param: str = "r") -> Any:
    """The ``k x k`` correlation matrix of whatever this is, as a float array:
    what every function here reads its matrix argument through.

    An array is returned as a float array; a mapping from
    :meth:`~polars_online.ModelBank.gram` or :func:`polars_online.gram.from_row`
    gives its ``comoments`` scaled; a one-row frame from
    :meth:`~polars_online.ModelBank.closed_groups` is read through
    :func:`polars_online.gram.from_row` first. ``ValueError`` for an array that is
    not square.
    """
    np = _np()
    if hasattr(obj, "to_dicts") or (
        isinstance(obj, dict) and "columns" in obj and "comoments" in obj
    ):
        from polars_online import gram as _gram

        g: Any = obj if isinstance(obj, dict) and "means" in obj else _gram.from_row(obj)
        return _gram.correlation(g)
    m = np.asarray(obj, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        msg = (
            f"{who}: {param} must be a square matrix, a Gram or a row of "
            f"ModelBank.closed_groups(), but it has shape {m.shape}. Pass a k by k correlation "
            "matrix, one of the Grams ModelBank.gram() returns, or a one-row frame of "
            "ModelBank.closed_groups()."
        )
        raise ValueError(msg)
    return m


@deprecated_alias(
    "po.corr.matrix",
    "po.gram.correlation",
    " for a Gram or a closed row's from_row, and every po.corr function takes the array, the "
    "Gram or the closed row itself",
)
def matrix(obj: Any) -> Any:
    """Renamed :func:`polars_online.gram.correlation` for a Gram: each call
    raises a :class:`~polars_online.PolarsOnlineDeprecationWarning`, then
    reads ``obj`` as every function here reads its matrix argument."""
    return _matrix(obj, "po.corr.matrix", "obj")


# --- Fisher's transform ------------------------------------------------------


def to_z(rho: Any) -> Any:
    """Apply Fisher's transform, ``z = atanh(rho)``, to correlations elementwise.

    Use it to move correlations to a scale on which their sampling error is
    about normal, with variance ``1 / (n - 3)``, before you average,
    difference or test them. :func:`from_z` maps back.

    .. rubric:: Parameters

    ``rho``
        A correlation, or an array of them. ``|rho|`` is clipped at
        :data:`Z_CLIP`, ``1 - 1e-6``, so a degenerate ``±1`` gives a finite
        ``z`` of about 7.25, where ``atanh`` would give ``inf``.

    .. rubric:: Returns

    A float, or a float array of ``rho``'s shape.

    .. rubric:: Examples

    .. code-block:: python

        print(po.corr.to_z([0.0, 0.5, 1.0]).round(3))

    .. code-block:: text

        [0.    0.549 7.254]

    .. rubric:: See also

    - :func:`from_z`, the inverse.
    - :func:`fisher_se`, the standard error of ``z``.

    .. rubric:: Notes

    Because of the clip, the round trip through :func:`from_z` is not quite
    the identity at the ends: it gives back ``±Z_CLIP``, to rounding, for
    ``±1``.
    """
    np = _np()
    return np.arctanh(np.clip(np.asarray(rho, dtype=float), -Z_CLIP, Z_CLIP))


def from_z(z: Any) -> Any:
    """Map Fisher's ``z`` back to a correlation, ``tanh(z)``, elementwise.

    Use it to return an average, a difference or a bound computed on the
    ``z`` scale to the correlation scale. It is the inverse of :func:`to_z`
    away from that function's clip.

    .. rubric:: Parameters

    ``z``
        A value of ``z``, or an array of them.

    .. rubric:: Returns

    A float, or a float array of ``z``'s shape, each in ``[-1, 1]``.

    .. rubric:: Examples

    .. code-block:: python

        z = po.corr.to_z([0.2, 0.6])
        print(po.corr.from_z(z.mean()).round(3))     # the mean correlation on the z scale

    .. code-block:: text

        0.42

    .. rubric:: See also

    - :func:`to_z`, the transform.
    """
    np = _np()
    return np.tanh(np.asarray(z, dtype=float))


# --- repair ------------------------------------------------------------------


def nearest(
    a: Any, w: Any = None, *, tol: float = 1e-8, max_iter: int = 100
) -> tuple[Any, float, int]:
    """Find the nearest correlation matrix to a matrix, by Higham's alternating projections.

    Use it to repair an "almost-correlation" matrix, one that is not
    positive semidefinite, such as one assembled from estimates of each
    pair made apart, while changing it as little as possible. To trade bias
    for variance instead, use :func:`shrink`.

    .. rubric:: Parameters

    ``a``
        A square, finite matrix, symmetrized on the way in; or a Gram from
        :meth:`~polars_online.ModelBank.gram`, or a row of
        :meth:`~polars_online.ModelBank.closed_groups`. **A Gram is repaired
        without its intercept column**, the constant a spec's
        ``fit_intercept`` adds, since a constant has no correlation with
        anything. It is read as :func:`polars_online.gram.correlation` reads
        it, and :func:`polars_online.gram.vif` leaves the intercept out the
        same way. A feature the stream found constant has
        no correlation either, and is refused: pass the Gram without it, as
        :func:`polars_online.gram.subset` returns it.
    ``w``
        The diagonal of a weight matrix ``W``, as a vector of one weight per
        column (for a Gram, per column other than the intercept), each
        finite and above 0. Default ``None``, every weight 1.
    ``tol``
        The stopping rule: the largest of the relative changes in ``X``, in
        ``Y``, and between them falls below it (Notes). Default 1e-8.
        ``tol`` sets the work, since convergence is linear.
    ``max_iter``
        The most iterations, at least 1. Default 100. It stops a run that
        has not met ``tol``, and returns the last iterate with no warning:
        ``iters < max_iter`` in the result shows that the run met ``tol``,
        and ``iters == max_iter`` that it may have stopped at the limit.
        At 0 the input would come back unprojected with a distance of 0,
        which would say it was already a correlation matrix.

    .. rubric:: Returns

    ``(X, dist, iters)``: the repaired matrix, the weighted Frobenius
    distance ``||W^1/2 (A - X) W^1/2||_F``, and the iterations taken. For a
    Gram, ``X`` has one row and one column for each of the Gram's columns
    other than the intercept, in their order. ``X`` has an exactly unit diagonal
    and is positive semidefinite to ``tol``: its smallest eigenvalue can be
    a small negative number of that order (Notes).

    .. rubric:: Raises

    ``ValueError``
        An entry of ``a`` that is not finite; a Gram column, other than the
        intercept, with no variance; ``a`` not square; ``w`` without one
        weight per column, or with a weight that is not finite and above 0;
        ``max_iter`` below 1.

    .. rubric:: Examples

    .. code-block:: python

        almost = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.3], [0.7, 0.3, 1.0]])   # not PSD
        fixed, dist, iters = po.corr.nearest(almost)
        print(fixed.round(3), round(dist, 4), iters)

    .. code-block:: text

        [[1.    0.895 0.697]
         [0.895 1.    0.303]
         [0.697 0.303 1.   ]] 0.0097 17

    .. rubric:: See also

    - :func:`shrink`, the other repair, toward a structured target.

    .. rubric:: Notes

    **The method is Higham's (2002) Algorithm 3.3**: alternating projections
    with Dykstra's correction,

    .. code-block:: text

        dS_0 = 0,  Y_0 = A
        R_k  = Y_{k-1} - dS_{k-1}          # Dykstra's correction
        X_k  = P_S(R_k)                    # onto the positive semidefinite cone
        dS_k = X_k - R_k
        Y_k  = P_U(X_k)                    # onto the unit diagonal

    with ``P_U`` setting the diagonal to 1 and, for the diagonal weight
    ``W``, ``P_S(A) = W^-1/2 (W^1/2 A W^1/2)_+ W^-1/2``, where ``(.)_+``
    clips the eigenvalues at 0. It stops on Higham's test (4.1): the
    largest of the relative changes in ``X``, in ``Y``, and between them,
    below ``tol``, each in the infinity norm, the largest absolute row sum.

    **Convergence is linear**: the change fell by a factor of 2.4 an
    iteration on Higham's 3 x 3 example and 2.7 on his 4 x 4, from
    ``tol=1e-4`` to ``1e-10``.

    **The result is positive semidefinite only to** ``tol``. The iteration
    converges to the boundary of the cone, so the smallest eigenvalue can
    be a small negative number of the order of ``tol``. Tighten ``tol`` if
    that matters: clipping it here would break the unit diagonal again.
    """
    np = _np()
    is_gram = hasattr(a, "to_dicts") or (
        isinstance(a, dict) and "columns" in a and "comoments" in a
    )
    cols: list[str] = []
    keep: list[int] = []
    if is_gram:
        # A Gram's intercept is a constant column, whose correlations are
        # nan: read it without, as po.gram.vif does by default (task 352).
        from polars_online import gram as _gram

        g: Any = a if isinstance(a, dict) and "means" in a else _gram.from_row(a)
        cols = _gram._columns(g, "po.corr.nearest")
        keep = [i for i, c in enumerate(cols) if c != _gram.INTERCEPT]
        x = _gram.correlation(g)[np.ix_(keep, keep)]
    else:
        x = _matrix(a, "po.corr.nearest", "a")
    k = x.shape[0]
    # A NaN propagates through the eigendecomposition and comes back as a
    # matrix of NaNs with a NaN distance -- an answer, and a wrong one
    # (docs/REVIEW-E54-E64.md K1).
    if not np.all(np.isfinite(x)):
        dead = [cols[keep[i]] for i in range(k) if is_gram and not np.isfinite(x[i, i])]
        if dead:
            which = (
                f"{dead[0]!r} has" if len(dead) == 1 else f"{', '.join(repr(c) for c in dead)} have"
            )
            msg = (
                "po.corr.nearest: every column of a's Gram but the intercept must vary, so "
                f"that it has a correlation, but {which} no variance. Pass the Gram without "
                "those columns, as po.gram.subset(g, columns) returns it."
            )
        else:
            msg = (
                "po.corr.nearest: every entry of a must be finite, but one is NaN or infinite. "
                "Replace the entries that are not finite before the repair."
            )
        raise ValueError(msg)
    if max_iter < 1:
        msg = (
            f"po.corr.nearest: max_iter must be at least 1, but it is {max_iter}. "
            "Set max_iter to 1 or more."
        )
        raise ValueError(msg)
    x = 0.5 * (x + x.T)
    if w is None:
        wv = np.ones(k)
    else:
        wv = np.asarray(w, dtype=float).reshape(-1)
        if wv.shape != (k,):
            msg = (
                f"po.corr.nearest: w must hold {k} weights, one for each column, but it "
                f"holds {wv.size}. Pass {k} weights, or leave w out to weigh every column "
                "alike."
            )
            raise ValueError(msg)
        # An infinite weight passed `> 0` and made the projection's scaling
        # inf * 0 = nan (task 352).
        bad = np.flatnonzero(~(np.isfinite(wv) & (wv > 0.0)))
        if bad.size:
            msg = (
                "po.corr.nearest: every weight in w must be finite and above 0, but "
                f"w[{int(bad[0])}] is {float(wv[bad[0]])!r}. Pass finite weights above 0, or "
                "leave w out to weigh every column alike."
            )
            raise ValueError(msg)
    root = np.sqrt(wv)
    inv_root = 1.0 / root

    def project_psd(m: Any) -> Any:
        scaled = root[:, None] * m * root[None, :]
        vals, vecs = np.linalg.eigh(0.5 * (scaled + scaled.T))
        clipped = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
        return inv_root[:, None] * clipped * inv_root[None, :]

    def project_unit(m: Any) -> Any:
        out = m.copy()
        np.fill_diagonal(out, 1.0)
        return out

    def norm(m: Any) -> float:
        # The infinity norm, the largest absolute row sum, as Higham's (4.1)
        # has it; the largest entry stopped his 4x4 a step after his 19
        # (review round 4, YB7).
        return float(np.abs(m).sum(axis=1).max())

    def rel(p: Any, q: Any) -> float:
        d = norm(q)
        return norm(p - q) / d if d > 0.0 else 0.0

    ds = np.zeros((k, k))
    y = x.copy()
    x_prev = y.copy()
    y_prev = y.copy()
    xk = y.copy()
    iters = 0
    for iters in range(1, max_iter + 1):  # noqa: B007 - the count is returned
        r = y - ds
        xk = project_psd(r)
        ds = xk - r
        y = project_unit(xk)
        if max(rel(xk, x_prev), rel(y, y_prev), rel(y, xk)) < tol:
            break
        x_prev, y_prev = xk.copy(), y.copy()
    # `Y` is the algorithm's answer: the iterate with the unit diagonal.
    # It is PSD only to `tol` -- the iteration converges to the boundary of
    # the cone, and how close is what `tol` buys -- so its smallest
    # eigenvalue can be a small negative number of that order. The
    # docstring says so; clipping it here would break the unit diagonal
    # again, and the caller who needs strict PSD wants a smaller `tol`.
    out = y
    diff = root[:, None] * (x - out) * root[None, :]
    return out, float(np.linalg.norm(diff, "fro")), iters


def shrink(
    r: Any, target: str = "constant", alpha: float | None = None, x: Any = None
) -> tuple[Any, float]:
    """Shrink a correlation matrix toward a structured target, by Ledoit and Wolf.

    Use it to make a noisy sample correlation matrix better conditioned,
    trading bias for variance: ``(1 - alpha) R + alpha F``. It repairs a
    sample matrix from fewer rows than columns, which is singular, in one
    step. To change the matrix as little as possible instead, use
    :func:`nearest`.

    .. rubric:: Parameters

    ``r``
        The matrix ``R``: a ``k x k`` correlation or covariance matrix, or a
        Gram or a closed group's row, read as
        :func:`polars_online.gram.correlation` reads it.
    ``target``
        The target ``F``. ``"constant"``, the default, is Ledoit and Wolf's
        constant-correlation target ("Honey, I shrunk the sample covariance
        matrix"): ``f_ii = r_ii`` and ``f_ij = rbar * sqrt(r_ii r_jj)``, with
        ``r_ij`` the entries of ``R`` and ``rbar`` the mean off-diagonal
        correlation. On a correlation matrix
        it is the equicorrelation matrix at ``rbar``. ``"identity"`` is their
        scaled identity ("A well-conditioned estimator for large-dimensional
        covariance matrices"): ``f = mu I``, with ``mu`` the mean of the
        diagonal. On a correlation matrix it is the identity.
    ``alpha``
        The intensity, from 0 (no shrinkage) to 1 (the target alone).
        Default ``None``, Ledoit and Wolf's optimal intensity, which needs
        ``x`` (Notes). One of ``alpha`` and ``x`` is required.
    ``x``
        The ``T x k`` sample the matrix came from, at least two rows, for
        the optimal intensity: the rows standardized for a correlation
        matrix, and as they are for a covariance matrix. **The sample has to be
        the one the matrix is of** (Notes). Pass ``alpha`` yourself when the
        two cannot be the same sample.

    .. rubric:: Returns

    ``(shrunk, alpha)``: the ``k x k`` shrunk matrix, and the intensity
    used. The result is positive definite whenever the target is and
    ``alpha`` is above 0.

    .. rubric:: Raises

    ``ValueError``
        A ``target`` that is neither word; ``alpha`` outside ``[0, 1]``;
        neither ``x`` nor ``alpha``; an ``x`` of the wrong shape or with
        fewer than two rows; ``r`` an array that is not square.

    .. rubric:: Examples

    .. code-block:: python

        rng = np.random.default_rng(0)
        x = rng.standard_normal((20, 30))                 # fewer rows than columns
        r = np.corrcoef(x, rowvar=False)                  # singular
        shrunk, alpha = po.corr.shrink(r, x=(x - x.mean(0)) / x.std(0))
        print(round(alpha, 3), np.linalg.eigvalsh(shrunk).min() > 0)

    .. code-block:: text

        0.867 True

    .. rubric:: See also

    - :func:`nearest`, the repair that changes the matrix least.
    - :func:`equicorr`, the ``rbar`` of the constant-correlation target.

    .. rubric:: Notes

    **The optimal intensity** is Ledoit and Wolf's (2004),
    ``alpha = max(0, min(kappa / T, 1))``, with
    ``kappa = (pi - rho) / gamma``. With ``x_it`` the value of column ``i``
    in row ``t`` of ``x``, and ``xbar_i`` that column's mean, for both
    targets:

    .. code-block:: text

        pi_ij    = (1/T) sum_t ((x_it - xbar_i)(x_jt - xbar_j) - s_ij)**2
        pi       = sum_ij pi_ij
        gamma    = sum_ij (f_ij - s_ij)**2

    and ``rho``, the target's own term, for the constant correlation:

    .. code-block:: text

        theta_ii,ij = (1/T) sum_t ((x_it - xbar_i)**2 - s_ii)
                              * ((x_it - xbar_i)(x_jt - xbar_j) - s_ij)
        rho      = sum_i pi_ii + sum_{i!=j} (rbar/2)
                     * (sqrt(s_jj/s_ii) theta_ii,ij + sqrt(s_ii/s_jj) theta_jj,ij)

    and for the identity, whose diagonal is ``mu`` rather than ``s_ii``:

    .. code-block:: text

        rho      = 0

    which is ``sklearn.covariance.ledoit_wolf``'s intensity. ``pi`` and
    ``theta`` are sums of fourth moments, so they cannot be recovered from
    ``R`` and ``T``: that is why the optimal intensity needs ``x``.

    **The sample has to be the one the matrix is of.** ``s_ij`` above is the
    plain covariance of ``x``, dividing by ``T``, and ``gamma`` compares the target with it.
    So an ``r`` that is another estimate of the same pairs, such as a
    decayed one out of a bank, makes ``kappa`` a comparison of two different
    matrices. The intensity is still in ``[0, 1]`` and the result still a
    shrunk matrix, but it is not the optimal intensity for the ``r`` passed.

    Shrinkage and :func:`nearest` answer different questions: shrinkage
    trades bias for variance, and Higham's projection changes the matrix as
    little as possible.
    """
    np = _np()
    s = _matrix(r, "po.corr.shrink")
    k = s.shape[0]
    off = ~np.eye(k, dtype=bool)
    sd = np.sqrt(np.clip(np.diag(s), 0.0, None))
    if target == "constant":
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = s / np.outer(sd, sd)
        rbar = float(np.nanmean(corr[off])) if k > 1 else 0.0
        f = rbar * np.outer(sd, sd)
        np.fill_diagonal(f, np.diag(s))
    elif target == "identity":
        rbar = 0.0
        f = np.eye(k) * np.mean(np.diag(s))
    else:
        msg = (
            f'po.corr.shrink: target must be "constant" or "identity", but it is {target!r}. '
            "Set target to one of those words."
        )
        raise ValueError(msg)

    if alpha is None:
        if x is None:
            msg = (
                "po.corr.shrink: neither x nor alpha is given, and the shrinkage intensity "
                "cannot be estimated from the matrix alone, since it reads the rows' fourth "
                "moments. Pass x, the sample of rows by columns, or set alpha."
            )
            raise ValueError(msg)
        xa = np.asarray(x, dtype=float)
        if xa.ndim != 2 or xa.shape[1] != k:
            msg = (
                f"po.corr.shrink: x must have one column for each of the matrix's {k} "
                f"columns, but it has shape {xa.shape}. Pass the sample as a 2-d array of "
                f"rows by {k} columns."
            )
            raise ValueError(msg)
        t = xa.shape[0]
        if t < 2:
            msg = (
                f"po.corr.shrink: x needs at least 2 rows to estimate the shrinkage intensity, "
                f"but it has {t}. Pass more rows, or set alpha."
            )
            raise ValueError(msg)
        d = xa - xa.mean(axis=0)
        # `s_ij` as the paper defines it, over the same rows.
        sam = d.T @ d / t
        prod = d[:, :, None] * d[:, None, :]  # (T, k, k)
        pi_ij = ((prod - sam) ** 2).mean(axis=0)
        pi = float(pi_ij.sum())
        if target == "identity":
            # Ledoit and Wolf (2004, JMVA): the target's diagonal is `mu`, so
            # it shares no estimation error with the sample's, and there is
            # no `rho`. The constant-correlation target's `sum_i pi_ii` was
            # kept here, 16-36% below scikit-learn (review round 4, YB4).
            rho = 0.0
        else:
            var = np.diagonal(prod, axis1=1, axis2=2)  # (T, k)
            # theta[i, j] = theta_ii,ij
            theta = ((var[:, :, None] - np.diag(sam)[None, :, None]) * (prod - sam)).mean(axis=0)
            sd_s = np.sqrt(np.clip(np.diag(sam), 0.0, None))
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.outer(1.0 / sd_s, sd_s)  # ratio[i, j] = sqrt(s_jj / s_ii)
            cross = (rbar / 2.0) * (ratio * theta + ratio.T * theta.T)
            rho = float(np.diag(pi_ij).sum() + np.nan_to_num(cross)[off].sum())
        gamma = float(((f - sam) ** 2).sum())
        alpha = 0.0 if gamma <= 0.0 else max(0.0, min((pi - rho) / gamma / t, 1.0))
    if not 0.0 <= alpha <= 1.0:
        msg = (
            f"po.corr.shrink: alpha must be from 0 to 1, but it is {alpha}. "
            "Set alpha from 0 (no shrinkage) to 1 (the target alone)."
        )
        raise ValueError(msg)
    return (1.0 - alpha) * s + alpha * f, float(alpha)


# --- summaries ---------------------------------------------------------------


def equicorr(r: Any) -> float:
    """Compute the mean off-diagonal correlation of a matrix.

    Use it to summarize a correlation matrix by one common correlation,
    the number a ``deco`` model tracks. For the same number from one
    standardized row, use :func:`equicorr_row`. For a common correlation per
    block, use :func:`block_means`.

    .. rubric:: Parameters

    ``r``
        A ``k x k`` correlation matrix, or a Gram or a closed group's row,
        read as :func:`polars_online.gram.correlation` reads it. ``nan``
        entries, such as a constant column's, are left out of the mean.

    .. rubric:: Returns

    A float: the mean of the off-diagonal entries. It is ``nan`` below two
    columns, and where every off-diagonal entry is ``nan``, with numpy's
    ``RuntimeWarning``.

    .. rubric:: Raises

    ``ValueError``
        ``r`` is an array that is not square.

    .. rubric:: Examples

    .. code-block:: python

        r = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
        print(round(po.corr.equicorr(r), 3))

    .. code-block:: text

        0.3

    .. rubric:: See also

    - :func:`equicorr_row` and :func:`equicorr_loglik`, the same model on one
      row.
    - :func:`block_means`, a mean correlation per block.
    """
    np = _np()
    m = _matrix(r, "po.corr.equicorr")
    k = m.shape[0]
    if k < 2:
        return float("nan")
    off = ~np.eye(k, dtype=bool)
    return float(np.nanmean(m[off]))


def equicorr_row(row: Any) -> float:
    """Estimate the common correlation from one standardized row, by Engle and Kelly's closed form.

    Use it to compute, offline, the per-row estimate a ``deco`` model reads:
    Engle and Kelly's Lemma 2.3. For the mean correlation of a whole matrix,
    use :func:`equicorr`.

    .. rubric:: Parameters

    ``row``
        One row of ``n`` standardized values (each series less its mean,
        over its standard deviation).

    .. rubric:: Returns

    A float:

    .. code-block:: text

        u = (S1**2 - S2) / ((n - 1) * S2),    S1 = sum(x),  S2 = sum(x * x)

    with ``x`` the row. It is ``nan`` below two entries or at ``S2 = 0``.

    .. rubric:: Examples

    .. code-block:: python

        print(round(po.corr.equicorr_row([1.2, 0.8, 1.0, -0.2]), 3))

    .. code-block:: text

        0.504

    .. rubric:: See also

    - :func:`equicorr`, the mean correlation of a matrix.
    - :func:`equicorr_loglik`, the row's log-density at a common
      correlation.
    """
    np = _np()
    v = np.asarray(row, dtype=float).reshape(-1)
    n = v.size
    if n < 2:
        return float("nan")
    s1, s2 = float(v.sum()), float((v * v).sum())
    return (s1 * s1 - s2) / ((n - 1) * s2) if s2 > 0.0 else float("nan")


def equicorr_loglik(row: Any, rho: float) -> float:
    """Compute the Gaussian log-density of a standardized row under a common correlation.

    Use it to score, offline, a row against an equicorrelation matrix, the
    ``loglik`` a ``deco`` model writes. To estimate the common correlation
    from the row, use :func:`equicorr_row`.

    .. rubric:: Parameters

    ``row``
        One row of ``n`` standardized values.
    ``rho``
        The common correlation, every off-diagonal entry of the matrix
        ``R``. Inside ``(-1/(n-1), 1)`` ``R`` is positive definite.

    .. rubric:: Returns

    A float, in closed form with ``S1 = sum(x)`` and ``S2 = sum(x * x)`` over
    the row ``x``:

    .. code-block:: text

        det R    = (1 - rho)**(n-1) * (1 + (n-1) rho)
        x'R^-1 x = (S2 - rho S1**2 / (1 + (n-1) rho)) / (1 - rho)
        loglik   = -0.5 * (n log(2 pi) + log det R + x'R^-1 x)

    It is ``nan`` below two entries, and outside ``(-1/(n-1), 1)``, where
    ``R`` is not positive definite.

    .. rubric:: Examples

    .. code-block:: python

        row = [1.2, 0.8, 1.0, -0.2]
        independent = po.corr.equicorr_loglik(row, 0.0)
        common = po.corr.equicorr_loglik(row, 0.4)
        print(round(independent, 3), round(common, 3))

    .. code-block:: text

        -5.236 -4.716

    .. rubric:: See also

    - :func:`equicorr_row`, the row's own estimate of the common
      correlation.
    """
    np = _np()
    v = np.asarray(row, dtype=float).reshape(-1)
    n = v.size
    if n < 2 or not (-1.0 / (n - 1) < rho < 1.0):
        return float("nan")
    s1, s2 = float(v.sum()), float((v * v).sum())
    log_det = (n - 1) * np.log1p(-rho) + np.log1p((n - 1) * rho)
    quad = (s2 - rho * s1 * s1 / (1.0 + (n - 1) * rho)) / (1.0 - rho)
    return float(-0.5 * (n * np.log(2 * np.pi) + log_det + quad))


def absorption(r: Any, k: int) -> float:
    """Compute Kritzman, Li, Page and Rigobon's absorption ratio of a correlation matrix.

    Use it to measure how much the series move as one: the share of the
    total variance the top ``k`` eigenvectors explain. A high ratio means
    the market is moving as one thing. To compare a fast and a slow ratio,
    use :func:`absorption_shift`.

    .. rubric:: Parameters

    ``r``
        An ``n x n`` correlation matrix, or a Gram or a closed group's
        row, read as :func:`polars_online.gram.correlation` reads it.
    ``k``
        How many of the largest eigenvalues to sum, from 1 to ``n``.

    .. rubric:: Returns

    A float, ``sum_{i<=k} lam_i / sum_i lam_i`` over the eigenvalues
    ``lam_i`` in decreasing order, or ``nan`` where they sum to 0.

    .. rubric:: Raises

    ``ValueError``
        ``k`` outside ``1..n``, or ``r`` an array that is not square.
    ``numpy.linalg.LinAlgError``
        ``r`` holds ``nan`` entries where numpy's eigensolver does not
        converge, as a Gram with an ``intercept`` column does (the module's
        docstring says why).

    .. rubric:: Examples

    .. code-block:: python

        r = np.array([[1.0, 0.8, 0.7], [0.8, 1.0, 0.75], [0.7, 0.75, 1.0]])
        print(round(po.corr.absorption(r, 1), 3))

    .. code-block:: text

        0.834

    .. rubric:: See also

    - :func:`absorption_shift`, a fast ratio against a slow one.
    - :func:`spectral`, the eigenpairs themselves.
    """
    np = _np()
    vals = np.linalg.eigvalsh(_matrix(r, "po.corr.absorption"))[::-1]
    total = float(vals.sum())
    if not 1 <= k <= vals.size:
        msg = (
            f"po.corr.absorption: k must be from 1 to the matrix's {vals.size} columns, but it "
            f"is {k}. Set k to the number of leading eigenvalues to sum, from 1 to {vals.size}."
        )
        raise ValueError(msg)
    return float(vals[:k].sum() / total) if total > 0.0 else float("nan")


def absorption_shift(ar_fast: Any, ar_slow: Any, *, scale: float | None = None) -> Any:
    """Compute Kritzman, Li, Page and Rigobon's standardized absorption shift.

    Use it to flag a rise in how much the series move as one: a fast
    absorption ratio against a slow one, in units of the slow one's spread.
    Compute each ratio with :func:`absorption` over windows of your choice.

    .. rubric:: Parameters

    ``ar_fast``, ``ar_slow``
        Two series of absorption ratios aligned row by row, the first over a
        short window and the second over a long one.
    ``scale``
        The divisor. Default ``None``, the population standard deviation of
        ``ar_slow`` over the sample (dividing by the number of rows).

    .. rubric:: Returns

    A float array, ``(ar_fast - ar_slow) / scale`` elementwise, or ``nan``
    throughout where ``scale`` is not above 0.

    .. rubric:: Raises

    ``ValueError``
        ``ar_fast`` and ``ar_slow`` of different shapes.

    .. rubric:: Examples

    .. code-block:: python

        fast = np.array([0.60, 0.62, 0.70, 0.80])
        slow = np.array([0.60, 0.61, 0.63, 0.66])
        print(po.corr.absorption_shift(fast, slow).round(2))

    .. code-block:: text

        [0.   0.44 3.06 6.11]

    .. rubric:: See also

    - :func:`absorption`, the ratio.

    .. rubric:: Notes

    The function's older name, ``shift``, is refused, and the error names
    this one: Polars uses ``shift`` for a lag.
    """
    np = _np()
    fast = np.asarray(ar_fast, dtype=float)
    slow = np.asarray(ar_slow, dtype=float)
    if fast.shape != slow.shape:
        msg = (
            f"po.corr.absorption_shift: ar_fast and ar_slow must have the same shape, but they "
            f"have {fast.shape} and {slow.shape}. Pass two series aligned row by row."
        )
        raise ValueError(msg)
    s = float(np.std(slow)) if scale is None else float(scale)
    return (fast - slow) / s if s > 0.0 else np.full_like(fast, np.nan)


#: Renamed :func:`absorption_shift` before 1.0, since Polars' ``shift`` is a
#: lag: the old name is refused, and the error names the new one.
shift = renamed_function("po.corr.shift", "po.corr.absorption_shift")


def spectral(r: Any, k: int) -> tuple[Any, Any]:
    """Compute the top ``k`` eigenpairs of a matrix, largest first.

    Use it to read a correlation matrix's leading factors, or to keep them
    and rebuild the matrix with :func:`from_spectral`. For the share of the
    variance they explain, use :func:`absorption`.

    .. rubric:: Parameters

    ``r``
        An ``n x n`` symmetric matrix, or a Gram or a closed group's
        row, read as :func:`polars_online.gram.correlation` reads it.
    ``k``
        How many eigenpairs to keep, a whole number of at least 1. A ``k``
        above ``n`` keeps all ``n``.

    .. rubric:: Returns

    ``(values, vectors)``: the ``k`` largest eigenvalues in decreasing
    order, and their eigenvectors as the rows of a ``k x n`` array, each
    signed so that its entry of largest magnitude is positive.

    .. rubric:: Raises

    ``ValueError``
        ``r`` is an array that is not square, or ``k`` is below 1.
    :class:`~polars_online.WrongTypeError`
        ``k`` is not a whole number, such as ``2.5``. It is both a
        ``TypeError`` and a ``ValueError``.
    ``numpy.linalg.LinAlgError``
        ``r`` holds ``nan`` entries where numpy's eigensolver does not
        converge, as a Gram with an ``intercept`` column does (the module's
        docstring says why).

    .. rubric:: Examples

    .. code-block:: python

        r = np.array([[1.0, 0.8, 0.7], [0.8, 1.0, 0.75], [0.7, 0.75, 1.0]])
        values, vectors = po.corr.spectral(r, 1)
        print(values.round(3), vectors.round(3))

    .. code-block:: text

        [2.501] [[0.578 0.59  0.564]]

    .. rubric:: See also

    - :func:`from_spectral`, the matrix rebuilt from the pairs.
    - :func:`absorption`, the share of the variance the top ``k`` explain.

    .. rubric:: Notes

    The sign rule is the one ``ew_cov``'s ``pca`` uses on its first refresh.
    Later refreshes keep each vector's sign continuous with the refresh
    before, and ``pca`` reads the covariance, not the correlation, so its
    vectors differ from these, beyond the sign, unless every variance is
    equal.
    """
    np = _np()
    wrong_type = isinstance(k, bool) or not isinstance(k, numbers.Integral)
    if wrong_type or operator.index(k) < 1:
        msg = (
            f"po.corr.spectral: k must be a whole number of at least 1, but it is {k!r}. Set k "
            "to the number of eigenpairs to keep, 1 or more."
        )
        raise WrongTypeError(msg) if wrong_type else ValueError(msg)
    k = operator.index(k)
    m = _matrix(r, "po.corr.spectral")
    vals, vecs = np.linalg.eigh(m)
    order = np.argsort(vals)[::-1][:k]
    out_vals = vals[order]
    out_vecs = vecs[:, order].T.copy()
    for row in out_vecs:
        j = int(np.argmax(np.abs(row)))
        if row[j] < 0.0:
            row *= -1.0
    return out_vals, out_vecs


def from_spectral(vals: Any, vecs: Any, *, unit_diag: bool = True) -> Any:
    """Rebuild a matrix from eigenpairs, completed to a unit diagonal.

    Use it to turn the leading factors :func:`spectral` keeps into a
    correlation matrix, such as a denoised one that keeps only the
    eigenvalues above :func:`mp_edge`'s noise edge.

    .. rubric:: Parameters

    ``vals``
        The eigenvalues, a vector of ``k``.
    ``vecs``
        The eigenvectors as the rows of a ``k x n`` array, as
        :func:`spectral` returns them.
    ``unit_diag``
        ``True``, the default, adds ``1 - diag(V' diag(vals) V)``, with ``V``
        the rows of ``vecs``, to the diagonal, so
        the result is a correlation matrix. ``False`` returns
        ``V' diag(vals) V`` as it is.

    .. rubric:: Returns

    An ``n x n`` float array, ``V' diag(vals) V`` with ``V`` the rows of
    ``vecs``, its diagonal set to 1 under ``unit_diag``.

    .. rubric:: Raises

    ``ValueError``
        ``vecs`` without one row per entry of ``vals``.

    .. rubric:: Examples

    .. code-block:: python

        r = np.array([[1.0, 0.8, 0.7], [0.8, 1.0, 0.75], [0.7, 0.75, 1.0]])
        top = po.corr.from_spectral(*po.corr.spectral(r, 1))   # the one-factor matrix
        print(top.round(3))

    .. code-block:: text

        [[1.    0.852 0.815]
         [0.852 1.    0.832]
         [0.815 0.832 1.   ]]

    .. rubric:: See also

    - :func:`spectral`, the eigenpairs.
    - :func:`mp_edge`, the noise edge to keep eigenvalues above.

    .. rubric:: Notes

    The remainder ``1 - diag(V' diag(vals) V)`` is not negative when
    ``vals`` are a correlation matrix's leading eigenvalues, because the
    dropped components are positive semidefinite. So adding it to the
    diagonal leaves a correlation matrix, not something that only looks
    like one.
    """
    np = _np()
    lam = np.asarray(vals, dtype=float).reshape(-1)
    v = np.asarray(vecs, dtype=float)
    if v.ndim != 2 or v.shape[0] != lam.size:
        msg = (
            f"po.corr.from_spectral: {lam.size} eigenvalues need {lam.size} eigenvectors, one "
            f"for each row of vecs, but vecs has shape {v.shape}. Pass vecs with one row for "
            "each entry of vals, as po.corr.spectral returns them."
        )
        raise ValueError(msg)
    m = v.T @ (lam[:, None] * v)
    if unit_diag:
        m = m + np.diag(1.0 - np.diag(m))
    return m


def block_means(r: Any, labels: Sequence[Any]) -> tuple[Any, Any]:
    """Compute the mean correlation within and between labeled blocks of columns.

    Use it to summarize a correlation matrix by block, such as by sector,
    and to test how far a block-equicorrelation matrix
    (:func:`from_blocks`) describes it. For one common correlation over
    every column, use :func:`equicorr`.

    .. rubric:: Parameters

    ``r``
        A ``k x k`` correlation matrix, or a Gram or a closed group's
        row, read as :func:`polars_online.gram.correlation` reads it.
    ``labels``
        One block label per column, of any type that compares equal.

    .. rubric:: Returns

    ``(B, counts)``, two arrays with one row and one column per block, in
    the order the labels first name the blocks. ``B[a, b]`` is the mean of
    ``r[i, j]`` over ``i`` in block ``a`` and ``j`` in block ``b``, leaving
    out the diagonal: a column's correlation with itself is 1 and says
    nothing about the block. ``counts[a, b]`` is the number of off-diagonal
    entries of ``r`` in the two blocks, and ``B[a, b]`` is ``nan`` where it
    is 0, as within a block of one column. ``nan`` entries of ``r`` are left
    out of the means but still counted, so a count can be larger than the
    number of entries behind its mean. Where every entry behind a mean is
    ``nan``, ``B[a, b]`` is ``nan`` too, and numpy issues a
    ``RuntimeWarning``.

    .. rubric:: Raises

    ``ValueError``
        ``labels`` without one label per column, or ``r`` an array that is
        not square.

    .. rubric:: Examples

    .. code-block:: python

        r = np.array([[1.0, 0.6, 0.1, 0.0],
                      [0.6, 1.0, 0.2, 0.1],
                      [0.1, 0.2, 1.0, 0.5],
                      [0.0, 0.1, 0.5, 1.0]])
        means, counts = po.corr.block_means(r, ["a", "a", "b", "b"])
        print(means, counts.tolist())

    .. code-block:: text

        [[0.6 0.1]
         [0.1 0.5]] [[2, 4], [4, 2]]

    .. rubric:: See also

    - :func:`from_blocks`, the matrix the block means describe.
    - :func:`equicorr`, one mean over every column.
    """
    np = _np()
    m = _matrix(r, "po.corr.block_means")
    lab = list(labels)
    if len(lab) != m.shape[0]:
        msg = (
            f"po.corr.block_means: labels must name a block for each of the matrix's "
            f"{m.shape[0]} columns, but it holds {len(lab)} labels. Pass one label for each column."
        )
        raise ValueError(msg)
    names: list[Any] = []
    for value in lab:
        if value not in names:
            names.append(value)
    idx = [np.array([i for i, v in enumerate(lab) if v == name]) for name in names]
    nb = len(names)
    b = np.full((nb, nb), np.nan)
    counts = np.zeros((nb, nb), dtype=np.int64)
    for a in range(nb):
        for c in range(nb):
            block = m[np.ix_(idx[a], idx[c])]
            mask = np.ones(block.shape, dtype=bool)
            if a == c:
                np.fill_diagonal(mask, False)
            counts[a, c] = int(mask.sum())
            if counts[a, c]:
                b[a, c] = float(np.nanmean(block[mask]))
    return b, counts


def from_blocks(b: Any, labels: Sequence[Any]) -> Any:
    """Build the block-equicorrelation matrix that a table of block correlations describes.

    Use it to turn the block means :func:`block_means` returns, or a table
    of your own, back into a full matrix: a structured target or a
    forecast. It has a unit diagonal, but only some tables give a positive
    semidefinite one. Repair one that is not with :func:`nearest`.

    .. rubric:: Parameters

    ``b``
        The block correlations, one row and one column per block, in the
        order ``labels`` first names the blocks. It must be symmetric, to
        within 1e-12, the rounding of two block means summed in different
        orders; ``nan`` entries match each other.
    ``labels``
        One block label per column of the result.

    .. rubric:: Returns

    A ``k x k`` float array, with ``k`` the number of labels. Entry
    ``(i, j)`` is ``B[block(i), block(j)]`` off the diagonal, and 1 on it,
    with ``B`` the table ``b``.

    .. rubric:: Raises

    ``ValueError``
        ``b`` not square in the number of blocks the labels name, or not
        symmetric.

    .. rubric:: Examples

    .. code-block:: python

        b = np.array([[0.6, 0.1], [0.1, 0.5]])
        print(po.corr.from_blocks(b, ["a", "a", "b", "b"]))

    .. code-block:: text

        [[1.  0.6 0.1 0.1]
         [0.6 1.  0.1 0.1]
         [0.1 0.1 1.  0.5]
         [0.1 0.1 0.5 1. ]]

    .. rubric:: See also

    - :func:`block_means`, its inverse up to the averaging within blocks.
    """
    np = _np()
    bm = np.asarray(b, dtype=float)
    lab = list(labels)
    names: list[Any] = []
    for value in lab:
        if value not in names:
            names.append(value)
    if bm.shape != (len(names), len(names)):
        msg = (
            f"po.corr.from_blocks: labels name {len(names)} blocks, so b must be a "
            f"{len(names)} by {len(names)} matrix, but it has shape {bm.shape}. Pass one row "
            "and one column of b for each block, in the order the labels first name them."
        )
        raise ValueError(msg)
    if not np.allclose(bm, bm.T, rtol=0.0, atol=1e-12, equal_nan=True):
        i, j = np.unravel_index(int(np.nanargmax(np.abs(bm - bm.T))), bm.shape)
        msg = (
            f"po.corr.from_blocks: b must be symmetric, but b[{i}, {j}] is {bm[i, j]:.3g} and "
            f"b[{j}, {i}] is {bm[j, i]:.3g}. Pass a symmetric table, such as (b + b.T) / 2."
        )
        raise ValueError(msg)
    of = np.array([names.index(v) for v in lab])
    m = bm[np.ix_(of, of)]
    np.fill_diagonal(m, 1.0)
    return m


# --- the noise floor ---------------------------------------------------------


def mp_edge(n: int, m: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Compute the Marchenko-Pastur edges for ``n`` observations of ``m`` series.

    Use it to tell a correlation matrix's eigenvalues from noise:
    eigenvalues between the edges are what pure noise produces, and only
    those above the upper edge carry information (Laloux, Cizeau, Bouchaud
    and Potters, 1999). For the density between the edges, use
    :func:`mp_density`.

    .. rubric:: Parameters

    ``n``
        The number of observations, above 0.
    ``m``
        The number of series, above 0.
    ``sigma2``
        The noise variance. Default 1, a correlation matrix's.

    .. rubric:: Returns

    ``(lam_minus, lam_plus)``, two floats:

    .. code-block:: text

        lam_pm = sigma2 * (1 +- 1/sqrt(Q))**2,    Q = n / m

    ``Q = 1`` gives ``(0, 4 sigma2)``.

    .. rubric:: Raises

    ``ValueError``
        ``n`` or ``m`` not above 0.

    .. rubric:: Examples

    .. code-block:: python

        lo, hi = po.corr.mp_edge(n=400, m=100)
        print(round(lo, 3), round(hi, 3))

    .. code-block:: text

        0.25 2.25

    .. rubric:: See also

    - :func:`mp_density`, the curve between the edges.
    - :func:`spectral`, the eigenvalues to compare with them.
    """
    return _mp_edge(n, m, sigma2, "po.corr.mp_edge")


def _mp_edge(n: int, m: int, sigma2: float, who: str) -> tuple[float, float]:
    """:func:`mp_edge`, its refusal naming ``who``, the call the reader made."""
    if n <= 0 or m <= 0:
        msg = (
            f"{who}: n and m must be above 0, but they are {n} and {m}. "
            "Set n to the number of rows and m to the number of columns."
        )
        raise ValueError(msg)
    q = n / m
    root = (1.0 / q) ** 0.5
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def mp_density(lam: Any, n: int, m: int, sigma2: float = 1.0) -> Any:
    """Compute the Marchenko-Pastur density at given eigenvalues.

    Use it to draw a correlation matrix's spectrum against the curve pure
    noise would give. For the edges alone, use :func:`mp_edge`.

    .. rubric:: Parameters

    ``lam``
        An eigenvalue, or an array of them.
    ``n``, ``m``, ``sigma2``
        As for :func:`mp_edge`.

    .. rubric:: Returns

    A float array of ``lam``'s shape, 0 outside the edges:

    .. code-block:: text

        (Q / (2 pi sigma2)) * sqrt((lam_plus - lam) * (lam - lam_minus)) / lam

    with ``Q``, ``lam_plus`` and ``lam_minus`` those of :func:`mp_edge`.

    .. rubric:: Raises

    ``ValueError``
        As for :func:`mp_edge`.

    .. rubric:: Examples

    .. code-block:: python

        print(po.corr.mp_density([0.1, 1.0, 3.0], n=400, m=100).round(3))

    .. code-block:: text

        [0.    0.616 0.   ]

    .. rubric:: See also

    - :func:`mp_edge`, the edges.
    """
    np = _np()
    lo, hi = _mp_edge(n, m, sigma2, "po.corr.mp_density")
    x = np.asarray(lam, dtype=float)
    q = n / m
    inside = (x > lo) & (x < hi) & (x > 0.0)
    out = np.zeros_like(x)
    with np.errstate(invalid="ignore"):
        out = np.where(
            inside,
            q / (2.0 * np.pi * sigma2) * np.sqrt(np.clip((hi - x) * (x - lo), 0.0, None)) / x,
            0.0,
        )
    return out


def signal_share(z_blocks: Any, n_kish_blocks: Any) -> Any:
    """Estimate how much of a correlation's movement across blocks is not sampling noise.

    Use it to judge whether a correlation measured in each block, such as
    each week, really moves, or moves no more than its sampling error would
    make it. A share of 0 means the correlation moved no more than noise
    would.

    .. rubric:: Parameters

    ``z_blocks``
        The Fisher ``z`` (:func:`to_z`) of each block's correlation: a
        vector over ``B`` blocks for one pair, or a ``B x p`` array for
        ``p`` pairs.
    ``n_kish_blocks``
        Each block's sample size, a vector of ``B``: Kish's effective sample
        size ``(sum w)**2 / sum w**2``, as
        :meth:`~polars_online.ModelBank.gram`'s ``n_kish`` reports it, or a
        plain block's row count. Not ``weight_sum``, which is a weight,
        about half the Kish size in a steady state.

    .. rubric:: Returns

    The share, per pair, with ``z_b`` and ``n_b`` block ``b``'s:

    .. code-block:: text

        clip(1 - mean_b(1 / (n_b - 3)) / var_b(z_b), 0, 1)

    A scalar for a 1-d ``z_blocks`` (one pair), else an array of ``p``. The
    share is ``nan`` where a block's sample size is 3 or less, whose ``z``
    has no such variance, and with fewer than two blocks.

    .. rubric:: Raises

    ``ValueError``
        ``n_kish_blocks`` without one size per block.

    .. rubric:: Examples

    .. code-block:: python

        z = po.corr.to_z(np.array([0.10, 0.35, 0.05, 0.40]))   # one pair's weekly correlations
        print(round(po.corr.signal_share(z, [200, 200, 200, 200]), 3))

    .. code-block:: text

        0.855

    .. rubric:: See also

    - :func:`fisher_se`, the sampling error of one correlation.

    .. rubric:: Notes

    ``1 / (n - 3)`` is the sampling variance of ``z``, so the ratio is the
    share of the observed variance across blocks that sampling noise
    explains, and one minus it is what is left.
    """
    np = _np()
    z = np.asarray(z_blocks, dtype=float)
    n = np.asarray(n_kish_blocks, dtype=float).reshape(-1)
    flat = z.ndim == 1
    zz = z.reshape(-1, 1) if flat else z
    if zz.shape[0] != n.size:
        msg = (
            f"po.corr.signal_share: n_kish_blocks must hold one sample size for each of the "
            f"{zz.shape[0]} blocks (rows) of z_blocks, but it holds {n.size}. Pass one sample "
            "size for each row of z_blocks."
        )
        raise ValueError(msg)
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = np.mean(np.where(n > 3.0, 1.0 / (n - 3.0), np.nan))
        observed = np.var(zz, axis=0, ddof=1) if zz.shape[0] > 1 else np.full(zz.shape[1], np.nan)
        share = np.clip(1.0 - floor / observed, 0.0, 1.0)
    return float(share[0]) if flat else share


# --- scoring a forecast ------------------------------------------------------


def loss(
    fcst: Any, real: Any, kind: str = "qlike", *, mu: Any = None, n: int | None = None
) -> float:
    """Score a forecast correlation matrix against the realized one.

    Use it to compare forecasts of a correlation or covariance matrix, such
    as two specs' ``ew_cov`` matrices against the matrix of the period that
    followed: the lower the loss, the better the forecast. Three losses are
    offered, each a proper scoring rule or a scaled error.

    .. rubric:: Parameters

    ``fcst``
        The forecast ``F``: a ``k x k`` matrix, or a Gram or a closed
        group's row, read as :func:`polars_online.gram.correlation` reads
        it.
    ``real``
        The realized matrix ``R``, in the same form and of the same shape.
    ``kind``
        ``"qlike"``, the default: the Gaussian quasi-likelihood loss,
        shifted by its forecast-free constant so that it is 0 at
        ``fcst == real`` and above 0 elsewhere. ``"z_mse"``: the squared
        Fisher-``z`` error in units of its own sampling variance, which needs
        ``n``. ``"minvar"``: Engle and Colacito's minimum-variance loss, the
        realized variance of the portfolio the forecast would have held.
    ``mu``
        For ``"minvar"``, the expected returns that set the portfolio.
        Default ``None``, a vector of ones.
    ``n``
        For ``"z_mse"``, the sample size behind the correlations.

    .. rubric:: Returns

    A float, by ``kind``:

    .. code-block:: text

        qlike   tr(F^-1 R) - log det(F^-1 R) - k
        z_mse   sum_{i<j} (z_ij(F) - z_ij(R))**2 * (n - 3)
        minvar  w' R w,    w = F^-1 mu / (mu' F^-1 mu)

    It is ``nan`` where the loss is undefined: a forecast with no inverse, a
    determinant of ``F^-1 R`` that is not above 0, or a zero denominator.

    .. rubric:: Raises

    ``ValueError``
        A ``kind`` that is none of the three; matrices of different shapes;
        ``"z_mse"`` without ``n``; an array that is not square.

    .. rubric:: Examples

    .. code-block:: python

        real = np.array([[1.0, 0.5], [0.5, 1.0]])
        good = np.array([[1.0, 0.4], [0.4, 1.0]])
        flat = np.eye(2)
        print(round(po.corr.loss(good, real), 4), round(po.corr.loss(flat, real), 4))

    .. code-block:: text

        0.0181 0.2877

    .. rubric:: See also

    - :func:`to_z`, the transform ``"z_mse"`` reads.

    .. rubric:: Notes

    ``"minvar"`` is smallest over ``F`` at ``F = R``, which is what makes it
    a proper scoring rule for a covariance forecast.
    """
    np = _np()
    f = _matrix(fcst, "po.corr.loss", "fcst")
    r = _matrix(real, "po.corr.loss", "real")
    if f.shape != r.shape:
        msg = (
            f"po.corr.loss: fcst and real must have the same shape, but they have {f.shape} "
            f"and {r.shape}. Pass two matrices of the same columns."
        )
        raise ValueError(msg)
    k = f.shape[0]
    if kind == "qlike":
        try:
            m = np.linalg.solve(f, r)
        except np.linalg.LinAlgError:
            # A forecast with no inverse, such as `nearest` returns on
            # Higham's own examples: undefined, as the docstring says, where
            # numpy's error came through (review round 4, YB13).
            return float("nan")
        sign, logdet = np.linalg.slogdet(m)
        if sign <= 0.0:
            return float("nan")
        return float(np.trace(m) - logdet - k)
    if kind == "z_mse":
        if n is None:
            msg = (
                'po.corr.loss: kind="z_mse" needs n, the sample size behind the correlations, '
                "but it is not set. Set n to the number of rows the correlations were computed "
                "from."
            )
            raise ValueError(msg)
        iu = np.triu_indices(k, 1)
        d = to_z(f[iu]) - to_z(r[iu])
        return float((d * d).sum() * (n - 3))
    if kind == "minvar":
        m = np.ones(k) if mu is None else np.asarray(mu, dtype=float).reshape(-1)
        try:
            w = np.linalg.solve(f, m)
        except np.linalg.LinAlgError:
            return float("nan")
        denom = float(m @ w)
        if denom == 0.0:
            return float("nan")
        w = w / denom
        return float(w @ r @ w)
    msg = (
        f'po.corr.loss: kind must be "qlike", "z_mse" or "minvar", but it is {kind!r}. '
        "Set kind to one of those words."
    )
    raise ValueError(msg)


# --- undoing the Epps attenuation --------------------------------------------


def epps_invert(gram_or_row: Any, *, L: int) -> Any:  # noqa: N803 - the paper's name
    """Compute the correlation over ``L`` rows from lagged co-moments, undoing the Epps attenuation.

    Use it on a correlation measured over fine intervals, which is too low
    because two series do not move at the same instants (the Epps effect).
    Toth and Kertesz's equation 12 undoes it by summing the lagged
    cross-covariances over a coarser interval of ``L`` rows.

    .. rubric:: Parameters

    ``gram_or_row``
        A Gram from :meth:`~polars_online.ModelBank.gram`, or a closed
        group's row, with ``lags`` and ``lag_comoments`` holding every lag
        ``1 .. L-1``. Build it with ``po.spec.ew_cov(..., lags=list(range(1,
        L)))``.
    ``L``
        The coarser interval, in rows, at least 1. ``L = 1`` gives the plain
        correlation.

    .. rubric:: Returns

    The ``k x k`` correlation matrix at scale ``L``, with ``nan`` where a
    column has no variance.

    .. rubric:: Raises

    ``ValueError``
        ``L`` below 1, or a lag ``1 .. L-1`` missing from the input, which
        the error names.

    .. rubric:: Examples

    This code reads ``trades`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        quotes = trades.filter(pl.col("price").is_null()).with_columns(
            ret=pl.col("mid").diff().over("symbol")
        )
        pair = quotes.pivot(on="symbol", index="ts", values="ret").fill_null(0.0)
        spec = po.spec.ew_cov("cov", features=["AAA", "BBB"], half_life=float("inf"), lags=[1, 2])
        bank = po.ModelBank([spec])
        bank.fit_predict(pair)
        g = bank.gram("cov")[0]
        print(round(float(po.corr.epps_invert(g, L=1)[0, 1]), 3),
              round(float(po.corr.epps_invert(g, L=3)[0, 1]), 3))

    .. code-block:: text

        -0.001 0.418

    .. rubric:: See also

    - :func:`polars_online.gram.correlation`, the correlation at one row.

    .. rubric:: Notes

    The weights are triangular, on the numerator and on both denominators:

    .. code-block:: text

        rho_L[a, b] = sum_x (L - |x|) C_x[a, b]
                      / sqrt(sum_x (L - |x|) C_x[a, a] * sum_x (L - |x|) C_x[b, b])

    over ``x = -(L-1) .. L-1``, with ``C_x`` the co-moments at lag ``x`` and
    ``C_{-x}[a, b] = C_x[b, a]``. So the numerator is ``L C_0[a, b] +
    sum_{l=1}^{L-1} (L - l) (C_l[a, b] + C_l[b, a])``, and each variance term
    is ``L C_0[a, a] + 2 sum_l (L - l) C_l[a, a]``.
    """
    np = _np()
    from polars_online import gram as _gram

    g = gram_or_row
    if not (isinstance(g, dict) and "comoments" in g):
        g = _gram.from_row(gram_or_row)
    if L < 1:
        msg = f"po.corr.epps_invert: L must be at least 1, but it is {L}. Set L to 1 or more."
        raise ValueError(msg)
    # Whatever form `gram()` handed it over in, packed or float32.
    c0 = _gram._comoments(np, g)
    k = c0.shape[0]
    lags = list(g.get("lags") or [])
    lag_c = g.get("lag_comoments")
    need = list(range(1, L))
    missing = [ell for ell in need if ell not in lags]
    if missing:
        msg = (
            f"po.corr.epps_invert: L = {L} needs the lags {need}, but lag {missing[0]} is not "
            f"in the input, which has {lags or 'none'}. Fit the ew_cov spec with "
            f"lags=list(range(1, {L})), or lower L."
        )
        raise ValueError(msg)
    num = float(L) * c0
    for ell in need:
        cl = np.asarray(lag_c[lags.index(ell)], dtype=float)
        num = num + (L - ell) * (cl + cl.T)
    auto = np.diag(num)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / np.sqrt(np.outer(auto, auto))
    out[~np.isfinite(out)] = np.nan
    np.fill_diagonal(out, np.where(auto > 0.0, 1.0, np.nan))
    assert out.shape == (k, k)
    return out


def fisher_se(
    n: float, rho: float | None = None, phi_a: float | None = None, phi_b: float | None = None
) -> float:
    """Compute the standard error of a correlation's Fisher ``z``, or of the correlation itself.

    Use it to judge whether a correlation differs from another, or from 0,
    by more than its sampling error, on Fisher's ``z`` scale or on the
    correlation's own. Give the two series' AR(1) coefficients to widen it
    for autocorrelated series.

    .. rubric:: Parameters

    ``n``
        The number of observations the correlation is estimated from.
    ``rho``
        Default ``None``, the standard error of Fisher's ``z``. Given, the
        delta-method error of the correlation ``rho`` itself, from -1 to 1.
    ``phi_a``, ``phi_b``
        The two series' AR(1) coefficients, given together or not at all.
        Given, the error is inflated for the pair's autocorrelation (Notes).

    .. rubric:: Returns

    A float:

    .. code-block:: text

        se = 1 / sqrt(n - 3)                                  # of z
           * (1 - rho**2)                                     # with rho
           * sqrt((1 + phi_a phi_b) / (1 - phi_a phi_b))      # with both phi

    It is ``nan`` at ``n <= 3``.

    .. rubric:: Raises

    ``ValueError``
        A ``rho`` outside ``[-1, 1]``, one ``phi`` without the other, since
        the inflation is a property of the pair, or ``phi_a * phi_b``
        outside ``(-1, 1)``, at any ``n``.

    .. rubric:: Examples

    .. code-block:: python

        print(round(po.corr.fisher_se(103), 4), round(po.corr.fisher_se(103, rho=0.5), 4),
              round(po.corr.fisher_se(103, phi_a=0.5, phi_b=0.5), 4))

    .. code-block:: text

        0.1 0.075 0.1291

    .. rubric:: See also

    - :func:`to_z`, the ``z`` this is the error of.
    - :func:`signal_share`, the same sampling error against a correlation's
      movement across blocks.

    .. rubric:: Notes

    The AR(1) inflation follows from Bartlett's formula, ``Var(r) ~ (1/n)
    sum_k rho_a(k) rho_b(k)``, and the geometric series ``1 + 2 sum_{k>=1}
    (phi_a phi_b)**k``. It holds under a true cross-correlation of 0 and
    linear dependence. It is not valid under ARCH-type innovations, where
    the variance of a sample correlation depends on the fourth moments and
    this understates it.
    """
    if rho is not None and not -1.0 <= rho <= 1.0:
        msg = (
            f"po.corr.fisher_se: rho must be a correlation, from -1 to 1, but it is {rho!r}. "
            "Set rho to the correlation whose error you want, or leave it out for the error "
            "of Fisher's z."
        )
        raise ValueError(msg)
    if (phi_a is None) != (phi_b is None):
        msg = (
            "po.corr.fisher_se: phi_a and phi_b must be given together, since the inflation "
            "reads both series, but only one is set. Set both, or neither."
        )
        raise ValueError(msg)
    p = None if phi_a is None or phi_b is None else phi_a * phi_b
    if p is not None and not -1.0 < p < 1.0:
        msg = (
            f"po.corr.fisher_se: the product of phi_a and phi_b must be above -1 and below "
            f"1, but it is {p:.3g}. Set each to an AR(1) coefficient above -1 and below 1."
        )
        raise ValueError(msg)
    if n <= 3:
        return float("nan")
    se = (n - 3.0) ** -0.5
    if rho is not None:
        se *= 1.0 - rho * rho
    if p is not None:
        se *= ((1.0 + p) / (1.0 - p)) ** 0.5
    return float(se)


def _label_unstable() -> None:
    """Every public function, labelled unstable:
    each call raises :class:`polars_online.UnstableWarning` under
    ``POLARS_ONLINE_WARN_UNSTABLE=1``, once however many of the others it
    runs. By name from ``__all__``, so a function added or renamed is
    labelled with the rest; a constant listed there is left as it is."""
    here = globals()
    for name in __all__:
        if callable(here[name]):
            here[name] = unstable(f"polars_online.corr.{name}")(here[name])


_label_unstable()
