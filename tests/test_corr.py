"""E62: `po.corr` — the arithmetic that comes after a correlation matrix.

Every function is held against a longhand check: Higham's own published
examples for `nearest`, Ledoit and Wolf's formulae written out for `shrink`,
and the closed forms for the rest. Where a value came from a paper it is
named in the test, so a disagreement is a finding rather than a mystery.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import corr

# Higham (2002) §4's two examples.
HIGHAM_3 = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
HIGHAM_3_NEAREST = np.array([[1.0, 0.7607, 0.1573], [0.7607, 1.0, 0.7607], [0.1573, 0.7607, 1.0]])
HIGHAM_4 = np.array(
    [
        [2.0, -1.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 2.0, -1.0],
        [0.0, 0.0, -1.0, 2.0],
    ]
)
HIGHAM_4_NEAREST = np.array(
    [
        [1.0, -0.8084, 0.1916, 0.1068],
        [-0.8084, 1.0, -0.6562, 0.1916],
        [0.1916, -0.6562, 1.0, -0.8084],
        [0.1068, 0.1916, -0.8084, 1.0],
    ]
)


def sample(n=400, k=4, rho=0.4, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 1))
    return np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.standard_normal((n, k))


# --- Fisher --------------------------------------------------------------


def test_the_transform_round_trips_and_the_clip_is_finite():
    r = np.array([-0.9, -0.2, 0.0, 0.5, 0.99])
    assert np.allclose(corr.from_z(corr.to_z(r)), r)
    assert np.isfinite(corr.to_z(1.0)) and corr.to_z(1.0) == pytest.approx(7.2543, abs=1e-3)
    assert corr.to_z(-1.0) == -corr.to_z(1.0)
    assert corr.to_z(0.5) == pytest.approx(np.arctanh(0.5))


# --- nearest -------------------------------------------------------------


def test_nearest_reproduces_highams_three_by_three():
    """§4: the matrix, the distance 0.5278, and a singular result whose null
    vector is `[-.4814, .7324, -.4814]`. The obvious guess `ee'` is at
    distance sqrt(2), which is what makes the example worth quoting."""
    x, dist, iters = corr.nearest(HIGHAM_3)
    assert np.allclose(x, HIGHAM_3_NEAREST, atol=1e-4)
    assert dist == pytest.approx(0.5278, abs=1e-3)
    vals, vecs = np.linalg.eigh(x)
    assert vals.min() == pytest.approx(0.0, abs=1e-7), "singular, as the paper says"
    null = vecs[:, 0] * np.sign(vecs[1, 0])
    assert np.allclose(np.abs(null), np.abs([-0.4814, 0.7324, -0.4814]), atol=1e-3)
    assert np.linalg.norm(HIGHAM_3 - np.ones((3, 3)), "fro") == pytest.approx(np.sqrt(2))
    assert iters > 1


def test_nearest_reproduces_highams_four_by_four_and_its_iteration_count():
    """§4's tridiagonal example: the matrix, distance 2.13, rank 3 -- and the
    iteration count, which pins the algorithm rather than just its fixed
    point. The paper reports 19 at `tol = 1e-8`; this counts the iteration
    in which the test passed, so the same run is 20 here."""
    x, dist, iters = corr.nearest(HIGHAM_4, tol=1e-8)
    assert np.allclose(x, HIGHAM_4_NEAREST, atol=1e-4)
    assert dist == pytest.approx(2.13, abs=1e-2)
    assert np.linalg.matrix_rank(x, tol=1e-7) == 3
    assert iters == 20


def test_nearest_obeys_highams_bounds():
    """Three consequences of his theorems, each a test."""
    # A diagonal matrix has nothing to repair but its scale.
    assert np.allclose(corr.nearest(np.diag([2.0, 3.0, 0.5]))[0], np.eye(3))
    # A PSD matrix with a diagonal at most 1 comes back with the diagonal
    # set to 1 and nothing else moved.
    a = np.array([[0.5, 0.2], [0.2, 0.4]])
    got, _, _ = corr.nearest(a)
    assert np.allclose(got, [[1.0, 0.2], [0.2, 1.0]], atol=1e-7)
    # Theorem 2.5: `t` nonpositive eigenvalues give at least `t` zero ones.
    t = int((np.linalg.eigvalsh(HIGHAM_3) <= 0).sum())
    zeros = int((np.abs(np.linalg.eigvalsh(corr.nearest(HIGHAM_3)[0])) < 1e-7).sum())
    assert t == 1 and zeros >= t


def test_nearest_is_a_correlation_matrix_on_random_inputs():
    rng = np.random.default_rng(1)
    for _ in range(30):
        k = int(rng.integers(2, 8))
        m = rng.standard_normal((k, k))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 1.0)
        x, dist, _ = corr.nearest(m, tol=1e-10)
        assert np.allclose(np.diag(x), 1.0), "an exactly unit diagonal"
        assert np.linalg.eigvalsh(x).min() > -1e-8, "PSD to the tolerance"
        assert dist <= np.linalg.norm(m - np.eye(k), "fro") + 1e-9


def test_a_correlation_matrix_is_returned_in_one_iteration():
    r = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    x, dist, iters = corr.nearest(r)
    assert iters == 1 and dist < 1e-12
    assert np.allclose(x, r)


def test_the_weights_change_which_entries_move():
    """`W` says whose entries matter: a heavily weighted variable's row
    moves less than an unweighted one's."""
    heavy = corr.nearest(HIGHAM_3, w=[100.0, 1.0, 1.0])[0]
    plain = corr.nearest(HIGHAM_3)[0]
    assert abs(heavy[0, 1] - 1.0) < abs(plain[0, 1] - 1.0)
    with pytest.raises(ValueError, match="positive weights"):
        corr.nearest(HIGHAM_3, w=[1.0, -1.0, 1.0])


# --- shrink --------------------------------------------------------------


def lw_alpha(x, rbar_from):
    """Ledoit and Wolf's intensity, their Appendices A-B written out."""
    t, k = x.shape
    d = x - x.mean(axis=0)
    s = d.T @ d / t
    sd = np.sqrt(np.diag(s))
    r = s / np.outer(sd, sd)
    off = ~np.eye(k, dtype=bool)
    rbar = r[off].mean() if rbar_from else 0.0
    f = rbar * np.outer(sd, sd)
    np.fill_diagonal(f, np.diag(s))
    prod = d[:, :, None] * d[:, None, :]
    pi_ij = ((prod - s) ** 2).mean(axis=0)
    pi = pi_ij.sum()
    var = np.diagonal(prod, axis1=1, axis2=2)
    theta = ((var[:, :, None] - np.diag(s)[None, :, None]) * (prod - s)).mean(axis=0)
    rho = np.diag(pi_ij).sum()
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            rho += (rbar / 2) * (
                np.sqrt(s[j, j] / s[i, i]) * theta[i, j] + np.sqrt(s[i, i] / s[j, j]) * theta[j, i]
            )
    gamma = ((f - s) ** 2).sum()
    return max(0.0, min((pi - rho) / gamma / t, 1.0)), f, s


def two_factor(n, seed):
    """Two blocks with different within-block correlations, so the
    constant-correlation target is *wrong* and the optimal intensity lands
    strictly inside (0, 1). A one-factor sample is exactly what the target
    describes, and there the intensity correctly saturates at 1."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 2))
    e = rng.standard_normal((n, 6))
    cols = [0.9 * f[:, 0] + 0.436 * e[:, i] for i in range(3)]
    cols += [0.3 * f[:, 1] + 0.954 * e[:, i + 3] for i in range(3)]
    return np.column_stack(cols)


def test_shrink_is_ledoit_and_wolfs_intensity():
    x = two_factor(120, 2)
    d = x - x.mean(axis=0)
    s = d.T @ d / len(x)
    want_alpha, want_f, _ = lw_alpha(x, rbar_from=True)
    got, alpha = corr.shrink(s, x=x)
    assert alpha == pytest.approx(want_alpha, rel=1e-12)
    assert np.allclose(got, (1 - alpha) * s + alpha * want_f)
    assert 0.0 < alpha < 1.0, f"a 120-row sample of 6 series shrinks partway, got {alpha}"


def test_shrink_saturates_where_the_target_is_right():
    """A one-factor sample *is* the constant-correlation target, so the
    optimal intensity is 1: nothing in the sample matrix is worth keeping
    over the structure."""
    x = sample(n=60, k=5, rho=0.3, seed=2)
    d = x - x.mean(axis=0)
    _, alpha = corr.shrink(d.T @ d / len(x), x=x)
    assert alpha == pytest.approx(1.0)


def test_shrink_at_the_ends_is_the_matrix_and_the_target():
    r = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]])
    assert np.allclose(corr.shrink(r, alpha=0.0)[0], r)
    target, _ = corr.shrink(r, alpha=1.0)
    rbar = corr.equicorr(r)
    want = np.full((3, 3), rbar)
    np.fill_diagonal(want, 1.0)
    assert np.allclose(target, want), "the constant-correlation target"
    ident, _ = corr.shrink(r, target="identity", alpha=1.0)
    assert np.allclose(ident, np.eye(3))


def test_shrink_repairs_a_singular_sample_matrix():
    """The reason it exists: fewer rows than columns gives a singular
    sample correlation matrix, and any positive intensity fixes it."""
    x = sample(n=6, k=10, seed=3)
    d = x - x.mean(axis=0)
    s = d.T @ d / len(x)
    assert np.linalg.eigvalsh(s).min() < 1e-12, "singular, as it must be"
    got, alpha = corr.shrink(s, x=x)
    assert alpha > 0.0
    assert np.linalg.eigvalsh(got).min() > 0.0


def test_shrink_needs_the_rows_or_the_intensity():
    r = np.eye(3)
    with pytest.raises(ValueError, match="fourth moment of the rows"):
        corr.shrink(r)
    with pytest.raises(ValueError, match="unknown target"):
        corr.shrink(r, target="nope", alpha=0.5)
    with pytest.raises(ValueError, match="alpha must be"):
        corr.shrink(r, alpha=1.5)


# --- summaries -----------------------------------------------------------


def test_equicorr_row_is_decos_u():
    """The offline pin on `deco`'s per-row estimate."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        r = rng.standard_normal(6)
        s1, s2 = r.sum(), (r * r).sum()
        assert corr.equicorr_row(r) == pytest.approx((s1**2 - s2) / (5 * s2))
    # And it is the mean off-diagonal product over the mean square.
    r = np.array([1.0, -0.5, 0.25])
    off = sum(r[i] * r[j] for i in range(3) for j in range(3) if i != j)
    assert corr.equicorr_row(r) == pytest.approx(off / (2 * (r * r).sum()))


def test_equicorr_loglik_is_the_dense_density():
    rng = np.random.default_rng(5)
    r = rng.standard_normal(5)
    for rho in (-0.1, 0.0, 0.4, 0.9):
        m = np.full((5, 5), rho)
        np.fill_diagonal(m, 1.0)
        sign, logdet = np.linalg.slogdet(m)
        want = -0.5 * (5 * np.log(2 * np.pi) + logdet + r @ np.linalg.solve(m, r))
        assert corr.equicorr_loglik(r, rho) == pytest.approx(want, rel=1e-10)
    assert np.isnan(corr.equicorr_loglik(r, 1.0))


def test_equicorr_is_the_mean_off_diagonal():
    r = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
    assert corr.equicorr(r) == pytest.approx((0.2 + 0.4 + 0.6) * 2 / 6)


def test_absorption_and_shift():
    x = sample(n=500, k=6, rho=0.7, seed=6)
    r = np.corrcoef(x.T)
    vals = np.linalg.eigvalsh(r)[::-1]
    assert corr.absorption(r, 1) == pytest.approx(vals[0] / vals.sum())
    assert corr.absorption(r, 6) == pytest.approx(1.0)
    assert corr.absorption(r, 1) > 0.6, "one factor, mostly one eigenvalue"
    fast = np.array([0.7, 0.8, 0.9])
    slow = np.array([0.6, 0.6, 0.7])
    assert np.allclose(corr.shift(fast, slow, scale=0.1), [1.0, 2.0, 2.0])
    assert np.allclose(corr.shift(fast, slow), (fast - slow) / np.std(slow))


def test_spectral_round_trips_through_from_spectral():
    x = sample(n=800, k=5, rho=0.5, seed=7)
    r = np.corrcoef(x.T)
    vals, vecs = corr.spectral(r, 5)
    assert np.allclose(corr.from_spectral(vals, vecs), r), "every component is the matrix"
    assert (vals[:-1] >= vals[1:]).all(), "descending"
    for row in vecs:
        assert row[np.argmax(np.abs(row))] > 0, "signed by the largest entry"
    # A truncation is still a correlation matrix.
    low = corr.from_spectral(*corr.spectral(r, 2))
    assert np.allclose(np.diag(low), 1.0)
    assert np.linalg.eigvalsh(low).min() > -1e-12


def test_block_means_round_trips_through_from_blocks():
    r = np.array(
        [
            [1.0, 0.8, 0.2, 0.1],
            [0.8, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.6],
            [0.1, 0.2, 0.6, 1.0],
        ]
    )
    labels = ["a", "a", "b", "b"]
    b, counts = corr.block_means(r, labels)
    assert b[0, 0] == pytest.approx(0.8) and b[1, 1] == pytest.approx(0.6)
    assert b[0, 1] == pytest.approx((0.2 + 0.1 + 0.3 + 0.2) / 4)
    assert counts.tolist() == [[2, 4], [4, 2]], "the diagonal is excluded"
    rebuilt = corr.from_blocks(b, labels)
    again, _ = corr.block_means(rebuilt, labels)
    assert np.allclose(again, b)
    with pytest.raises(ValueError, match="labels for a"):
        corr.block_means(r, ["a"])


# --- the noise floor ------------------------------------------------------


def test_mp_edges_and_density():
    assert corr.mp_edge(10, 10) == pytest.approx((0.0, 4.0))
    lo, hi = corr.mp_edge(400, 100, sigma2=2.0)
    q = 4.0
    assert lo == pytest.approx(2.0 * (1 - q**-0.5) ** 2)
    assert hi == pytest.approx(2.0 * (1 + q**-0.5) ** 2)
    # The density integrates to 1 between the edges.
    grid = np.linspace(lo, hi, 200_001)
    assert np.trapezoid(corr.mp_density(grid, 400, 100, 2.0), grid) == pytest.approx(1.0, abs=2e-3)
    assert corr.mp_density(np.array([hi * 1.5]), 400, 100)[0] == 0.0


def test_a_noise_spectrum_lies_inside_the_edges():
    rng = np.random.default_rng(8)
    n, m = 2000, 100
    x = rng.standard_normal((n, m))
    vals = np.linalg.eigvalsh(np.corrcoef(x.T))
    lo, hi = corr.mp_edge(n, m)
    # A finite-`n` margin: the edges are the limit, and the largest
    # eigenvalue exceeds them by O(n^-2/3).
    assert vals.max() < hi * 1.1 and vals.min() > lo * 0.5


def test_signal_share_is_zero_for_pure_noise_and_one_for_a_real_move():
    rng = np.random.default_rng(9)
    n = 200
    # Blocks whose true correlation never moves: all the variation is the
    # sampling floor.
    noise = np.array([np.arctanh(np.corrcoef(sample(n, 2, 0.5, s).T)[0, 1]) for s in range(40)])
    assert corr.signal_share(noise, np.full(40, n)) < 0.4
    # A correlation that swings from 0.1 to 0.8 is mostly signal.
    real = np.array(
        [np.arctanh(np.corrcoef(sample(n, 2, 0.1 if s % 2 else 0.8, s).T)[0, 1]) for s in range(40)]
    )
    assert corr.signal_share(real, np.full(40, n)) > 0.9
    # 2-D: one column per pair.
    both = np.column_stack([noise, real])
    got = corr.signal_share(both, np.full(40, n))
    assert got.shape == (2,) and got[1] > got[0]
    assert rng is not None


# --- scoring --------------------------------------------------------------


def test_qlike_is_zero_at_the_truth_and_positive_elsewhere():
    x = sample(n=500, k=4, rho=0.5, seed=10)
    r = np.corrcoef(x.T)
    assert corr.loss(r, r, "qlike") == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(11)
    for _ in range(20):
        f = corr.nearest(r + 0.05 * rng.standard_normal((4, 4)))[0]
        assert corr.loss(f, r, "qlike") > 0.0


def test_minvar_is_minimised_at_the_truth():
    """Engle and Colacito: the realised variance of the portfolio the
    forecast would have held is smallest when the forecast is right."""
    x = sample(n=800, k=4, rho=0.4, seed=12)
    r = np.corrcoef(x.T)
    best = corr.loss(r, r, "minvar")
    rng = np.random.default_rng(13)
    for _ in range(30):
        f = corr.nearest(r + 0.1 * rng.standard_normal((4, 4)))[0]
        assert corr.loss(f, r, "minvar") >= best - 1e-12


def test_z_mse_is_the_scaled_fisher_error():
    a = np.array([[1.0, 0.5], [0.5, 1.0]])
    b = np.array([[1.0, 0.2], [0.2, 1.0]])
    want = (np.arctanh(0.5) - np.arctanh(0.2)) ** 2 * (100 - 3)
    assert corr.loss(a, b, "z_mse", n=100) == pytest.approx(want)
    with pytest.raises(ValueError, match="needs `n`"):
        corr.loss(a, b, "z_mse")
    with pytest.raises(ValueError, match="unknown kind"):
        corr.loss(a, b, "nope")


# --- the Epps inversion ---------------------------------------------------


def test_epps_invert_recovers_a_lagged_pair():
    """`b` is `a` two rows late: the scale-1 correlation is near zero and
    the inverted one is near the truth."""
    n = 4000
    rng = np.random.default_rng(14)
    a = rng.standard_normal(n)
    b = np.concatenate([[0.0, 0.0], a[:-2]])
    df = pl.DataFrame({"x0": a, "x1": b})
    L = 8
    spec = po.spec.ew_cov("c", features=["x0", "x1"], lam=1.0, stats=[], lags=list(range(1, L)))
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    g = bank.gram("c")[0]
    plain = po.gram.correlation(g)[0, 1]
    assert abs(plain) < 0.1, "attenuated to nothing at scale 1"
    got = corr.epps_invert(g, L=L)
    assert got[0, 1] > 0.7, got[0, 1]
    assert np.allclose(np.diag(got), 1.0)
    # `L = 1` is the plain correlation.
    assert corr.epps_invert(g, L=1)[0, 1] == pytest.approx(plain, rel=1e-12)


def test_epps_invert_names_a_missing_lag():
    df = pl.DataFrame({"x0": [1.0, 2.0, 3.0], "x1": [2.0, 1.0, 4.0]})
    bank = po.ModelBank(
        [po.spec.ew_cov("c", features=["x0", "x1"], lam=1.0, stats=[], lags=[1, 3])]
    )
    bank.fit_predict(df)
    with pytest.raises(ValueError, match="lag 2 is not there"):
        corr.epps_invert(bank.gram("c")[0], L=4)
    with pytest.raises(ValueError, match="L must be >= 1"):
        corr.epps_invert(bank.gram("c")[0], L=0)


# --- standard errors ------------------------------------------------------


def test_fisher_se_is_its_closed_forms():
    assert corr.fisher_se(103) == pytest.approx(0.1)
    assert corr.fisher_se(103, rho=0.5) == pytest.approx(0.1 * 0.75)
    # The AR(1) inflation is Bartlett's sum, which the partial sum reaches.
    pa, pb = 0.8, 0.6
    partial = 1.0 + 2.0 * sum((pa * pb) ** k for k in range(1, 400))
    assert corr.fisher_se(103, phi_a=pa, phi_b=pb) == pytest.approx(0.1 * partial**0.5)
    assert corr.fisher_se(3) != corr.fisher_se(3)  # nan below the floor
    with pytest.raises(ValueError, match="go together"):
        corr.fisher_se(100, phi_a=0.5)
    with pytest.raises(ValueError, match=r"must be in \(-1, 1\)"):
        corr.fisher_se(100, phi_a=1.0, phi_b=1.0)


# --- the shared entry point -----------------------------------------------


def test_matrix_reads_an_array_a_gram_and_a_closed_row():
    df = pl.DataFrame({f"x{i}": v for i, v in enumerate(sample(300, 3, 0.5, 15).T)})
    df = df.with_columns(g=pl.int_range(pl.len()) // 150)
    spec = po.spec.ew_cov(
        "c", features=["x0", "x1", "x2"], lam=1.0, stats=[], group="g", group_close="monotone"
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    row = bank.closed_groups()
    from_row = corr.matrix(row)
    from_gram = corr.matrix(po.gram.from_row(row))
    assert np.allclose(from_row, from_gram)
    assert np.allclose(corr.matrix(from_row), from_row)
    assert np.allclose(np.diag(from_row), 1.0)
    with pytest.raises(ValueError, match="expected a square matrix"):
        corr.matrix(np.zeros((2, 3)))
