"""Correlation matrices, read and repaired (docs/ENHANCEMENTS.md E62).

:mod:`polars_online.gram`'s complement. Where that module solves and
diagnoses a design matrix, this one takes a **correlation matrix** and does
the arithmetic that comes after a pass: repair it to the nearest one that is
a correlation matrix, shrink it towards a structured target, summarise it as
an equicorrelation or a block matrix, place its eigenvalues against the
Marchenko-Pastur edges, score a forecast of it, undo the Epps attenuation,
and put a standard error on a single correlation.

Every function is a pure function of arrays. Where the input is a matrix it
may equally be a mapping from :meth:`~polars_online.ModelBank.gram` or a row
from :meth:`~polars_online.ModelBank.closed_groups`: :func:`matrix` reads
all three.

The arithmetic is the papers', named in each docstring, and every function
is held against a longhand check in ``tests/test_corr.py`` -- Higham's own
published examples for :func:`nearest`, Ledoit and Wolf's formulae for
:func:`shrink`, the closed forms for the rest.

Requires numpy, which is an optional extra of this package
(``pip install polars-online[numpy]``) -- not a dependency, as it is not one
of polars' either. Nothing here needs scipy or scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "absorption",
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
    "matrix",
    "mp_density",
    "mp_edge",
    "nearest",
    "shift",
    "shrink",
    "signal_share",
    "spectral",
    "to_z",
]

#: Fisher's transform is infinite at ±1, and a degenerate block gives
#: exactly that. Inputs are clipped here, so ``to_z(1.0)`` is about 7.25
#: rather than ``inf``; the round trip through :func:`from_z` is then not
#: quite the identity at the ends, which is the price of a finite number.
Z_CLIP = 1.0 - 1e-6


def _np() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as e:  # pragma: no cover - exercised by a stub
        msg = (
            "polars_online.corr works in numpy arrays, and numpy is not installed. "
            "Install it with `pip install numpy` or `pip install polars-online[numpy]`."
        )
        raise ModuleNotFoundError(msg) from e
    return np


def matrix(obj: Any) -> Any:
    """The ``k x k`` correlation matrix of whatever this is.

    An array (returned as a float array), a mapping from
    :meth:`~polars_online.ModelBank.gram` or
    :func:`polars_online.gram.from_row` (its ``comoments`` scaled), or a
    one-row frame from :meth:`~polars_online.ModelBank.closed_groups` (read
    through :func:`polars_online.gram.from_row` first).
    """
    np = _np()
    if hasattr(obj, "to_dicts") or (
        isinstance(obj, dict) and "columns" in obj and "comoments" in obj
    ):
        from polars_online import gram as _gram

        g = obj if isinstance(obj, dict) and "means" in obj else _gram.from_row(obj)
        return _gram.correlation(g)
    m = np.asarray(obj, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        msg = f"corr: expected a square matrix, a Gram mapping or a closed row; got {m.shape}"
        raise ValueError(msg)
    return m


# --- Fisher's transform ------------------------------------------------------


def to_z(rho: Any) -> Any:
    """Fisher's ``z = atanh(rho)``, elementwise, with ``|rho|`` clipped at
    :data:`Z_CLIP` so a degenerate ``±1`` is finite (``z ~ 7.25``)."""
    np = _np()
    return np.arctanh(np.clip(np.asarray(rho, dtype=float), -Z_CLIP, Z_CLIP))


def from_z(z: Any) -> Any:
    """``tanh(z)``, the inverse of :func:`to_z` away from the clip."""
    np = _np()
    return np.tanh(np.asarray(z, dtype=float))


# --- repair ------------------------------------------------------------------


def nearest(
    a: Any, w: Any = None, *, tol: float = 1e-8, max_iter: int = 100
) -> tuple[Any, float, int]:
    """The nearest correlation matrix to ``a``, by Higham (2002).

    Alternating projections with Dykstra's correction -- his Algorithm 3.3,
    written out::

        dS_0 = 0,  Y_0 = A
        R_k  = Y_{k-1} - dS_{k-1}          # Dykstra's correction
        X_k  = P_S(R_k)                    # onto the PSD cone
        dS_k = X_k - R_k
        Y_k  = P_U(X_k)                    # onto the unit diagonal

    with ``P_U`` setting the diagonal to 1 and, for a diagonal weight ``w``,
    ``P_S(A) = W^-1/2 (W^1/2 A W^1/2)_+ W^-1/2`` where ``(.)_+`` clips the
    eigenvalues at zero. It stops on his test (4.1): the largest of the
    relative infinity-norm changes in ``X``, in ``Y``, and between them,
    below ``tol``.

    Returns ``(X, dist, iters)`` with ``dist`` the weighted Frobenius
    distance ``||W^1/2 (A - X) W^1/2||_F``. Convergence is linear (a factor
    of about 3 an iteration on his own examples), so ``max_iter`` is a
    safety net rather than a knob; reaching it returns the last iterate.

    The result has an **exactly** unit diagonal and is PSD to ``tol``: the
    iteration converges to the boundary of the cone, so the smallest
    eigenvalue can be a small negative number of that order. Tighten
    ``tol`` if that matters; clipping it here would break the diagonal
    again.

    ``a`` must be square; it is symmetrised on the way in, since an
    "almost-correlation" matrix from two different estimates of the same
    pair is the case this exists for. ``w`` is the diagonal of ``W`` as a
    vector, all positive.
    """
    np = _np()
    x = matrix(a)
    k = x.shape[0]
    x = 0.5 * (x + x.T)
    if w is None:
        wv = np.ones(k)
    else:
        wv = np.asarray(w, dtype=float).reshape(-1)
        if wv.shape != (k,) or not np.all(wv > 0.0):
            msg = f"corr.nearest: w must be {k} positive weights, the diagonal of W"
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

    def rel(p: Any, q: Any) -> float:
        d = np.abs(q).max()
        return float(np.abs(p - q).max() / d) if d > 0.0 else 0.0

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
    """``(1 - a) R + a F``: Ledoit and Wolf (2004) shrinkage towards a
    structured target.

    ``target="constant"`` (the default) is their constant-correlation
    target: ``f_ii = s_ii`` and ``f_ij = rbar * sqrt(s_ii s_jj)`` with
    ``rbar`` the mean off-diagonal correlation, which on a correlation
    matrix is the equicorrelation matrix at ``rbar``.
    ``target="identity"`` is the identity.

    ``alpha`` fixes the intensity. Left out, it is their optimal
    ``delta = max(0, min(kappa / T, 1))`` with ``kappa = (pi - rho) / gamma``,
    which needs the **rows**::

        pi_ij    = (1/T) sum_t ((y_it - ybar_i)(y_jt - ybar_j) - s_ij)**2
        theta_ii,ij = (1/T) sum_t ((y_it - ybar_i)**2 - s_ii)
                              * ((y_it - ybar_i)(y_jt - ybar_j) - s_ij)
        rho      = sum_i pi_ii + sum_{i!=j} (rbar/2)
                     * (sqrt(s_jj/s_ii) theta_ii,ij + sqrt(s_ii/s_jj) theta_jj,ij)
        gamma    = sum_ij (f_ij - s_ij)**2

    ``pi`` and ``theta`` are fourth-moment sums, so they cannot be recovered
    from ``R`` and ``T``: pass ``x`` as the ``T x k`` sample the matrix came
    from (the standardised rows), or pass ``alpha`` yourself. One of the two
    is required.

    Returns ``(shrunk, alpha)``. The result is positive definite whenever
    the target is and ``alpha > 0``, which is the point: a sample
    correlation matrix from fewer rows than columns is singular, and this is
    the cheapest honest repair. :func:`nearest` is the other one, and they
    answer different questions -- shrinkage trades bias for variance,
    Higham's projection changes the matrix as little as possible.
    """
    np = _np()
    s = matrix(r)
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
        msg = f'corr.shrink: unknown target {target!r}; expected "constant" or "identity"'
        raise ValueError(msg)

    if alpha is None:
        if x is None:
            msg = (
                "corr.shrink: the optimal intensity is a fourth moment of the rows, which "
                "cannot be recovered from the matrix; pass `x` (the T x k sample) or `alpha`"
            )
            raise ValueError(msg)
        xa = np.asarray(x, dtype=float)
        if xa.ndim != 2 or xa.shape[1] != k:
            msg = f"corr.shrink: x must be T x {k}, got {xa.shape}"
            raise ValueError(msg)
        t = xa.shape[0]
        d = xa - xa.mean(axis=0)
        # `s_ij` as the paper defines it, over the same rows.
        sam = d.T @ d / t
        prod = d[:, :, None] * d[:, None, :]  # (T, k, k)
        pi_ij = ((prod - sam) ** 2).mean(axis=0)
        pi = float(pi_ij.sum())
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
        msg = f"corr.shrink: alpha must be in [0, 1], got {alpha}"
        raise ValueError(msg)
    return (1.0 - alpha) * s + alpha * f, float(alpha)


# --- summaries ---------------------------------------------------------------


def equicorr(r: Any) -> float:
    """The mean off-diagonal correlation: the one number a `deco` tracks."""
    np = _np()
    m = matrix(r)
    k = m.shape[0]
    if k < 2:
        return float("nan")
    off = ~np.eye(k, dtype=bool)
    return float(np.nanmean(m[off]))


def equicorr_row(row: Any) -> float:
    """Engle and Kelly's Lemma 2.3 on one **standardised** row::

        u = (S1**2 - S2) / ((n - 1) * S2),  S1 = sum(r), S2 = sum(r*r)

    the same closed form `deco` computes per row (ENHANCEMENTS E55), offline.
    """
    np = _np()
    v = np.asarray(row, dtype=float).reshape(-1)
    n = v.size
    if n < 2:
        return float("nan")
    s1, s2 = float(v.sum()), float((v * v).sum())
    return (s1 * s1 - s2) / ((n - 1) * s2) if s2 > 0.0 else float("nan")


def equicorr_loglik(row: Any, rho: float) -> float:
    """The Gaussian log-density of a **standardised** row under an
    equicorrelation matrix at ``rho``, in closed form::

        det R  = (1 - rho)**(n-1) * (1 + (n-1) rho)
        r'R^-1 r = (S2 - rho S1**2 / (1 + (n-1) rho)) / (1 - rho)

    which is `deco`'s ``loglik``, offline. ``nan`` outside
    ``(-1/(n-1), 1)``, where ``R`` is not a correlation matrix.
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
    """Kritzman, Li, Page and Rigobon's absorption ratio: the share of total
    variance the top ``k`` eigenvectors explain, ``sum_{i<=k} lam_i / sum
    lam_i``. High means the market is moving as one thing."""
    np = _np()
    vals = np.linalg.eigvalsh(matrix(r))[::-1]
    total = float(vals.sum())
    if not 1 <= k <= vals.size:
        msg = f"corr.absorption: k must be 1..{vals.size}, got {k}"
        raise ValueError(msg)
    return float(vals[:k].sum() / total) if total > 0.0 else float("nan")


def shift(ar_fast: Any, ar_slow: Any, *, scale: float | None = None) -> Any:
    """The standardised absorption shift ``(fast - slow) / scale``,
    elementwise over two aligned series of absorption ratios; ``scale``
    defaults to the standard deviation of ``ar_slow`` over the sample. The
    two windows are the caller's."""
    np = _np()
    fast = np.asarray(ar_fast, dtype=float)
    slow = np.asarray(ar_slow, dtype=float)
    if fast.shape != slow.shape:
        msg = f"corr.shift: the two series must align, got {fast.shape} and {slow.shape}"
        raise ValueError(msg)
    s = float(np.std(slow)) if scale is None else float(scale)
    return (fast - slow) / s if s > 0.0 else np.full_like(fast, np.nan)


def spectral(r: Any, k: int) -> tuple[Any, Any]:
    """The top ``k`` eigenpairs, descending: ``(values, vectors)`` with
    ``vectors`` row-major ``k x n``, each signed so its largest-magnitude
    entry is positive -- the rule `ew_cov`'s ``pca`` uses on a first
    refresh, so the two agree up to that convention."""
    np = _np()
    m = matrix(r)
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
    """``V' diag(vals) V``, completed to a unit diagonal when ``unit_diag``.

    The remainder ``1 - diag(V' L V)`` is non-negative, because the dropped
    components are PSD, so adding it to the diagonal leaves a correlation
    matrix rather than something that only looks like one.
    """
    np = _np()
    lam = np.asarray(vals, dtype=float).reshape(-1)
    v = np.asarray(vecs, dtype=float)
    if v.ndim != 2 or v.shape[0] != lam.size:
        msg = f"corr.from_spectral: {lam.size} values need {lam.size} vectors, got {v.shape}"
        raise ValueError(msg)
    m = v.T @ (lam[:, None] * v)
    if unit_diag:
        m = m + np.diag(1.0 - np.diag(m))
    return m


def block_means(r: Any, labels: Sequence[Any]) -> tuple[Any, Any]:
    """The mean correlation within and between labelled blocks.

    ``B[a, b]`` is the mean of ``R[i, j]`` over ``i`` in block ``a`` and
    ``j`` in block ``b``, **excluding the diagonal** (a variable's
    correlation with itself is 1 and says nothing about the block).
    Returns ``(B, counts)`` with the number of pairs behind each mean, in
    the order the labels first appear.
    """
    np = _np()
    m = matrix(r)
    lab = list(labels)
    if len(lab) != m.shape[0]:
        msg = f"corr.block_means: {len(lab)} labels for a {m.shape[0]}-column matrix"
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
    """The block-equicorrelation matrix ``B`` describes: entry ``(i, j)`` is
    ``B[block(i), block(j)]`` off the diagonal, and 1 on it. The inverse of
    :func:`block_means` up to the within-block averaging, which is the test.
    """
    np = _np()
    bm = np.asarray(b, dtype=float)
    lab = list(labels)
    names: list[Any] = []
    for value in lab:
        if value not in names:
            names.append(value)
    if bm.shape != (len(names), len(names)):
        msg = f"corr.from_blocks: {len(names)} blocks need a {len(names)}-square B, got {bm.shape}"
        raise ValueError(msg)
    of = np.array([names.index(v) for v in lab])
    m = bm[np.ix_(of, of)]
    np.fill_diagonal(m, 1.0)
    return m


# --- the noise floor ---------------------------------------------------------


def mp_edge(n: int, m: int, sigma2: float = 1.0) -> tuple[float, float]:
    """The Marchenko-Pastur edges for ``n`` observations of ``m`` series::

        lam_pm = sigma2 * (1 +- 1/sqrt(Q))**2,    Q = n / m

    (Laloux, Cizeau, Bouchaud and Potters 1999). Eigenvalues inside them are
    what pure noise produces; only those above ``lam_+`` carry information.
    ``Q = 1`` gives ``(0, 4 sigma2)``.
    """
    if n <= 0 or m <= 0:
        msg = f"corr.mp_edge: n and m must be positive, got {n} and {m}"
        raise ValueError(msg)
    q = n / m
    root = (1.0 / q) ** 0.5
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def mp_density(lam: Any, n: int, m: int, sigma2: float = 1.0) -> Any:
    """The Marchenko-Pastur density ``(Q / 2 pi sigma2) sqrt((lam_+ - lam)
    (lam - lam_-)) / lam``, zero outside the edges -- the curve to draw a
    spectrum against."""
    np = _np()
    lo, hi = mp_edge(n, m, sigma2)
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


def signal_share(z_blocks: Any, n_eff_blocks: Any) -> Any:
    """How much of the between-block movement in a correlation is not
    sampling noise.

    Per pair over ``B`` blocks, with ``z_b`` the Fisher-z of that block's
    correlation and ``n_b`` its effective sample size::

        clip(1 - mean_b(1 / (n_b - 3)) / var_b(z_b), 0, 1)

    ``1 / (n - 3)`` is the sampling variance of ``z``, so the ratio is the
    share of the observed variance the floor explains, and one minus it is
    what is left. ``0`` means the correlation moved no more than noise would.
    A 1-D input (one pair) returns a scalar.
    """
    np = _np()
    z = np.asarray(z_blocks, dtype=float)
    n = np.asarray(n_eff_blocks, dtype=float).reshape(-1)
    flat = z.ndim == 1
    zz = z.reshape(-1, 1) if flat else z
    if zz.shape[0] != n.size:
        msg = f"corr.signal_share: {zz.shape[0]} blocks of z against {n.size} sample sizes"
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
    """How wrong a forecast correlation matrix was.

    ``"qlike"`` is the Gaussian quasi-likelihood loss, shifted by the
    forecast-free constant so that it is **zero at ``fcst == real`` and
    positive elsewhere**::

        tr(F^-1 R) - log det(F^-1 R) - k

    ``"z_mse"`` is ``sum_{i<j} (z_ij(F) - z_ij(R))**2 * (n - 3)``, the
    squared Fisher-z error in units of its own sampling standard error;
    ``n`` is required.

    ``"minvar"`` is Engle and Colacito's minimum-variance loss ``w'Rw`` with
    ``w = F^-1 mu / (mu' F^-1 mu)``, the realised variance of the portfolio
    the forecast would have held; ``mu`` defaults to ones. It is minimised
    over ``F`` at ``F = R``, which is what makes it a proper scoring rule
    for a covariance forecast.
    """
    np = _np()
    f = matrix(fcst)
    r = matrix(real)
    if f.shape != r.shape:
        msg = f"corr.loss: the two matrices must match, got {f.shape} and {r.shape}"
        raise ValueError(msg)
    k = f.shape[0]
    if kind == "qlike":
        m = np.linalg.solve(f, r)
        sign, logdet = np.linalg.slogdet(m)
        if sign <= 0.0:
            return float("nan")
        return float(np.trace(m) - logdet - k)
    if kind == "z_mse":
        if n is None:
            msg = 'corr.loss: kind="z_mse" needs `n`, the sample size behind the correlations'
            raise ValueError(msg)
        iu = np.triu_indices(k, 1)
        d = to_z(f[iu]) - to_z(r[iu])
        return float((d * d).sum() * (n - 3))
    if kind == "minvar":
        m = np.ones(k) if mu is None else np.asarray(mu, dtype=float).reshape(-1)
        w = np.linalg.solve(f, m)
        denom = float(m @ w)
        if denom == 0.0:
            return float("nan")
        w = w / denom
        return float(w @ r @ w)
    msg = f'corr.loss: unknown kind {kind!r}; expected "qlike", "z_mse" or "minvar"'
    raise ValueError(msg)


# --- undoing the Epps attenuation --------------------------------------------


def epps_invert(gram_or_row: Any, *, L: int) -> Any:  # noqa: N803 - the paper's name
    """The correlation at scale ``L`` rows, from lagged co-moments at scale 1.

    Toth and Kertesz's equation 12: a correlation computed over fine
    intervals is attenuated because the two series do not move at the same
    instants, and the attenuation is undone by summing the lagged
    cross-covariances over the coarser interval. The weights are
    **triangular**, on the numerator and on both denominators::

        rho_L[a, b] = sum_x (L - |x|) C_x[a, b]
                      / sqrt(sum_x (L - |x|) C_x[a, a] * sum_x (L - |x|) C_x[b, b])

    over ``x = -(L-1) .. L-1`` with ``C_{-x}[a, b] = C_x[b, a]`` -- so the
    numerator is ``L C_0[a, b] + sum_{l=1}^{L-1} (L - l) (C_l[a, b] + C_l[b,
    a])`` and each auto term is ``L C_0[a, a] + 2 sum_l (L - l) C_l[a, a]``.

    The input is a mapping from :meth:`~polars_online.ModelBank.gram` or a
    closed row with ``lags`` and ``lag_comoments`` (ENHANCEMENTS E56), and
    must carry **every** lag ``1 .. L-1``: build it with
    ``ew_cov(lags=list(range(1, L)))``. A missing lag is an error naming it.

    ``L = 1`` is the plain correlation, which is the identity this reduces
    to.
    """
    np = _np()
    from polars_online import gram as _gram

    g = gram_or_row
    if not (isinstance(g, dict) and "comoments" in g):
        g = _gram.from_row(gram_or_row)
    if L < 1:
        msg = f"corr.epps_invert: L must be >= 1, got {L}"
        raise ValueError(msg)
    c0 = np.asarray(g["comoments"], dtype=float)
    k = c0.shape[0]
    lags = list(g.get("lags") or [])
    lag_c = g.get("lag_comoments")
    need = list(range(1, L))
    missing = [ell for ell in need if ell not in lags]
    if missing:
        msg = (
            f"corr.epps_invert: L = {L} needs lags {need}, and lag {missing[0]} is not there "
            f"(the input has {lags or 'none'}); accumulate with lags=list(range(1, {L}))"
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
    """The standard error of a correlation estimated from ``n`` observations.

    ``1 / sqrt(n - 3)`` is the standard error of Fisher's ``z``. With
    ``rho`` it is the delta-method error of the correlation itself,
    ``(1 - rho**2) / sqrt(n - 3)``.

    With both ``phi`` it is inflated by ``sqrt((1 + phi_a phi_b) / (1 -
    phi_a phi_b))`` for a pair of AR(1) series. That comes from Bartlett's
    formula ``Var(r) ~ (1/n) sum_k rho_a(k) rho_b(k)`` and the geometric
    series ``1 + 2 sum_{k>=1} (phi_a phi_b)**k``. **Both of its assumptions
    matter**: it holds under a *zero* true cross-correlation and linear
    dependence, and it is not valid under ARCH-type innovations, where the
    variance of a sample correlation depends on the fourth moments and this
    understates it.

    One ``phi`` without the other is an error: the inflation is a property
    of the pair.
    """
    if n <= 3:
        return float("nan")
    if (phi_a is None) != (phi_b is None):
        msg = "corr.fisher_se: phi_a and phi_b go together; the inflation is the pair's"
        raise ValueError(msg)
    se = (n - 3.0) ** -0.5
    if rho is not None:
        se *= 1.0 - rho * rho
    if phi_a is not None and phi_b is not None:
        p = phi_a * phi_b
        if not -1.0 < p < 1.0:
            msg = f"corr.fisher_se: phi_a * phi_b must be in (-1, 1), got {p}"
            raise ValueError(msg)
        se *= ((1.0 + p) / (1.0 - p)) ** 0.5
    return float(se)
