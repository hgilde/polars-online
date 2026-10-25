"""Convergence class 8: inner-iteration honesty (docs/PLAN.md task 278).

Wherever this library iterates inside a row -- a coordinate descent, a
Lloyd iteration, a refit -- it stops by a rule or by a cap. The class holds
two things on inputs built to make the iteration slow:

- **the stopping rule was met** where the library says it converged, checked
  against what the rule certifies: the lasso's KKT conditions, and its
  solution next to scikit-learn's (taken as given, not tested);
- **every cap hit is counted** where the library counts them, and where it
  does not, the silence is pinned in ``TestKnownDefects`` with the measured
  rate, so that the fix flips it.

What iterates, and what each says about a cap hit (2026-10-25, at
``fe98f01``):

==========================================  ===========================  =========
the iteration                               its stopping rule and cap    a cap hit
==========================================  ===========================  =========
``lasso``'s descent, per target and point   a sweep's largest move under  counted in
                                            ``tol``; ``max_iter`` sweeps  ``solve_failures``
``gram.lasso_path``, the same descent       the same; 1,000 sweeps        silent
``kmeans``' and ``hmm``'s ``lloyd`` seeding  a pass that assigns every    counted in
                                            row where the last did; 300  ``solve_failures``
                                            passes, from each of 10
                                            greedy k-means++ starts
``hmm``'s learning                          none needed: one filtered     --
                                            EM step a row, no inner loop
``audit``'s MAD bisection                   adjacent doubles; 2,100       cannot bind:
                                            halvings                      a double's
                                                                          exponents span
                                                                          2,098 halvings
``huber``'s and ``sgd(loss="huber")``'s     task 277's, registered when   --
warm-up refit                               it lands
==========================================  ===========================  =========
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Lasso

import polars_online as po
import polars_online.gram as gram
import reference_cluster as ref

TIER = "mixed"

U = 2.0**-53
FEATURES = ["x0", "x1", "x2"]


def gamma(m: int) -> float:
    return m * U / (1.0 - m * U)


def collinear(n: int, seed: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Two features `eps` apart and a third independent: the correlation
    matrix's condition is about `4 / eps^2`, and a cyclic coordinate
    descent contracts by about `1 - 1 / kappa` a sweep along the pair's
    difference, so it is slow by construction."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=(3, n))
    x = np.column_stack([2.0 + a, 2.0 + a + eps * b, -1.0 + c])
    return x, 0.7 + x @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=n)


def frame(x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({**{f: x[:, i] for i, f in enumerate(FEATURES)}, "y": y})


def failures(bank: po.ModelBank) -> int:
    return int(bank.solve_failures()["solve_failures"].sum())


def standardized_system(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """The lasso's problem as the model states it, written from the rows
    (half-life inf, unit weights): `C` the features' correlation matrix,
    `d_i = cov(x_i, y) / s_i`, `s` the population spreads and `zs` the
    standardized features; the objective is `b'Cb / 2 - d'b + l |b|_1`."""
    n = len(y)
    m = x.mean(0)
    s = np.sqrt(((x - m) ** 2).mean(0))
    zs = (x - m) / s
    return zs.T @ zs / n, zs.T @ (y - y.mean()) / n, s, zs


def kkt_violation(c: np.ndarray, d: np.ndarray, lam: float, b: np.ndarray) -> np.ndarray:
    """Each coordinate's distance from its KKT condition, the minimal-norm
    subgradient of the objective: `|g_i + l sign(b_i)|` where `b_i` is not
    zero and `max(|g_i| - l, 0)` where it is, `g = C b - d`."""
    g = c @ b - d
    return np.where(b != 0, np.abs(g + lam * np.sign(b)), np.maximum(np.abs(g) - lam, 0.0))


def last_solve(spec: dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """The coefficients after the last row, and whether that row's solve
    was counted unconverged: the rows but the last in one chunk, the last
    in another, `solve_failures` read between."""
    bank = po.ModelBank([spec])
    bank.fit_predict(frame(x, y)[:-1])
    before = failures(bank)
    out = bank.fit_predict(frame(x, y)[-1:])["m"].struct.unnest()
    return np.array(out["coef"][0], float), failures(bank) - before


def lasso_case(eps: float, lam: float, max_iter: int, tol: float, n: int = 800) -> dict[str, Any]:
    """One descent's honesty: was the last solve counted, and if not, is it
    within what its stopping rule certifies?"""
    x, y = collinear(n, 1, eps)
    spec = po.spec.lasso(
        "m",
        targets=["y"],
        features=FEATURES,
        penalties=[lam],
        half_life=float("inf"),
        coef_every=0,
        max_iter=max_iter,
        tol=tol,
    )
    coef, counted = last_solve(spec, x, y)
    c, d, s, zs = standardized_system(x, y)
    b = coef[1:] * s
    v = kkt_violation(c, d, lam, b)
    # The stopping rule: in the last sweep every coordinate moved by less
    # than `tol`. Coordinate `i` met its own condition exactly when it was
    # updated (the soft threshold is its exact minimizer); the coordinates
    # after it in the sweep then moved its gradient by at most `tol` each,
    # times `|C_ij|`. So the rule certifies `|v_i| <= tol * sum_j |C_ij|`.
    # The model's moments and these, written from the rows, differ by the
    # accumulation's rounding, `gamma_{4n}` of the terms of `C b` and `d`.
    certified = tol * np.abs(c).sum(1)
    rounding = 8 * gamma(4 * n) * (np.abs(c) @ np.abs(b) + np.abs(d) + lam)
    return {
        "x": x,
        "y": y,
        "b": b,
        "c": c,
        "d": d,
        "zs": zs,
        "counted": counted,
        "kkt": v,
        "bound": certified + rounding,
    }


#: The grid: the pair's distance (conditions from 4e2 to 4e8), the penalty,
#: and the descent's settings, the defaults and a strict pair.
GRID = [
    (eps, lam, max_iter, tol)
    for eps in (1e-1, 1e-2, 1e-3, 1e-4)
    for lam in (0.1, 0.01, 0.001, 0.0)
    for max_iter, tol in ((100, 1e-10), (100_000, 1e-14))
]


class TestLassoDescent:
    """The descent stops when a sweep moves no standardized coefficient by
    `tol`, or after `max_iter` sweeps, which `solve_failures` counts. A
    descent not counted must be within what its rule certifies, and a
    descent that is not within it must have been counted.

    Measured 2026-10-25 over `GRID`: every uncounted last solve sat at or
    below 1e-14 of KKT violation (bounds 2e-14 to 2e-10), every counted one
    at 1e-11 to 1e-5; the defaults count a cap hit on 798 of 799 solves at
    a penalty of 0.001 and a pair 0.1 apart (a condition of 4e2), and the
    strict settings still count every solve at 1e-4 (4e8): a cyclic descent
    there needs about `kappa` sweeps.
    """

    @staticmethod
    def check(eps, lam, max_iter, tol) -> dict[str, Any]:
        case = lasso_case(eps, lam, max_iter, tol)
        within = bool(np.all(case["kkt"] <= case["bound"]))
        print(
            f"eps {eps:g} l {lam:g} max_iter {max_iter} tol {tol:g}: counted {case['counted']},"
            f" KKT {case['kkt'].max():.1e} against {case['bound'].max():.1e}"
        )
        assert case["counted"] in (0, 1)
        if not case["counted"]:
            assert within, "a descent that stopped by its rule is not where the rule says"
        return case

    @pytest.mark.parametrize(
        "eps, lam, max_iter, tol",
        [(1e-3, 0.01, 100, 1e-10), (1e-3, 0.01, 100_000, 1e-14), (1e-1, 0.0, 100, 1e-10)],
    )
    def test_a_descent_either_meets_its_rule_or_is_counted(self, eps, lam, max_iter, tol):
        self.check(eps, lam, max_iter, tol)

    @pytest.mark.extended(reason="a grid: 32 descents and scikit-learn beside each")
    def test_every_descent_of_the_grid_meets_its_rule_or_is_counted(self):
        counted = [self.check(*point)["counted"] for point in GRID]
        assert 0 < sum(counted) < len(GRID), "both paths ran"

    @pytest.mark.parametrize("eps, lam", [(1e-2, 0.01), (1e-3, 0.1), (1e-4, 0.01)])
    def test_a_converged_descent_is_scikit_learns_lasso(self, eps, lam):
        # scikit-learn's `Lasso` on the same standardized rows, at the same
        # penalty, converged far past ours (its duality gap under 1e-14),
        # taken as given. The objective is strongly convex with modulus
        # `mu`, `C`'s least eigenvalue, so `|b - b*| <= |v| / mu` for the
        # minimal-norm subgradient `v`: the KKT violation bounds the
        # distance. scikit-learn's own distance is at most `sqrt(2 gap / mu)`.
        case = lasso_case(eps, lam, 100_000, 1e-14)
        assert not case["counted"], "the descent was meant to converge here"
        sk = Lasso(alpha=lam, fit_intercept=True, tol=1e-14, max_iter=10**7).fit(
            case["zs"], case["y"]
        )
        mu = float(np.linalg.eigvalsh(case["c"])[0])
        gap = 1e-14 * float(np.mean((case["y"] - case["y"].mean()) ** 2))
        bound = float(np.linalg.norm(case["bound"])) / mu + np.sqrt(2.0 * gap / mu)
        dist = float(np.linalg.norm(case["b"] - sk.coef_))
        print(f"eps {eps:g} l {lam:g}: |b - sklearn| {dist:.1e} against {bound:.1e}")
        assert dist <= bound

    def test_every_cap_hit_is_counted(self):
        # One sweep a descent and a `tol` no move can be under but an exact
        # 0: every descent whose sweep moved a coefficient is a cap hit. At
        # penalties every coefficient clears, on a stream whose fit moves on
        # every row, every solve moves every point, so the count is the
        # solves times the points: one solve a row under `half_life = inf`,
        # from the row whose weight first gives a fit (the second).
        x, y = collinear(300, 2, 0.3)
        points = [0.01, 0.001, 0.0]
        spec = po.spec.lasso(
            "m",
            targets=["y"],
            features=FEATURES,
            penalties=points,
            half_life=float("inf"),
            max_iter=1,
            tol=1e-300,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(frame(x, y))
        assert failures(bank) == len(points) * (len(y) - 1)


def eight_blobs(seed: int, n: int = 500) -> np.ndarray:
    """Eight unit blobs whose centres are drawn in a square of side 4, so
    they overlap: Lloyd's passes move rows between them for a while."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-2.0, 2.0, (8, 2))
    return centres[rng.integers(0, 8, n)] + rng.normal(size=(n, 2))


class TestLloydSeeding:
    """``init = "lloyd"``, the default seeding of ``kmeans`` and the one
    ``hmm`` shares, runs each of its ten greedy k-means++ starts to a fixed
    point -- a pass that assigns every row where the last did -- at most 300
    passes, and counts a run stopped there in ``solve_failures``. The recipe
    is ``tests/reference_cluster.py``'s, which ``test_kmeans.py`` holds to
    the library bit for bit. A run stopped at the cap is counted in the
    core's unit tests, which can lower it; no stream measured here reaches
    it.

    It ran ten passes from each start, with no test that the assignments had
    stopped changing and no count where they had not: on eight overlapping
    blobs in two dimensions (500 rows, ``kmeans``' default buffer), measured
    2026-10-25, 38 of 50 seeds kept centres that were not a fixed point,
    the largest move 0.12 spreads, and ``solve_failures`` read 0 (task 278,
    D7).
    """

    @pytest.mark.parametrize(
        "seeds",
        [
            5,
            pytest.param(
                50, marks=pytest.mark.extended(reason="a grid: 50 seedings in the Python reference")
            ),
        ],
    )
    def test_the_lloyd_seeding_ends_at_a_fixed_point(self, seeds):
        # One more pass from the kept centres assigns every row as the last
        # did and moves no centre, to the bit; and ten passes, the old rule,
        # stop short on these seeds, so the fixed point is not ten passes'.
        short, hits, at_ten = 0, 0, 0
        for seed in range(seeds):
            buf = eight_blobs(seed).tolist()
            w, mw = [1.0] * len(buf), [1.0, 1.0]
            kept, n = ref.seed_counted(buf, w, 8, "lloyd", seed, True, mw)
            assert kept is not None
            hits += n
            again, fixed = ref._lloyd(buf, w, [list(c) for c in kept], mw, 2)
            short += not (fixed and again == kept)
            at_ten += ref.seed_counted(buf, w, 8, "lloyd", seed, True, mw, cap=10)[1]
        print(
            f"lloyd: {short} of {seeds} short of a fixed point, {hits} runs at the cap; "
            f"{at_ten} of {10 * seeds} runs short at ten passes"
        )
        assert at_ten > 0, "ten passes reached every fixed point: the case is not slow"
        assert short == 0
        assert hits == 0
        # The library's own: seeded from the buffer alone, in the features'
        # units, its centres are the means of the rows its seeds own, which
        # for seeds at a fixed point are the seeds, so one more pass from
        # them moves no row. Rows past the buffer's far cut (five times its
        # mean squared distance from the mean, in two dimensions) choose no
        # seed but are replayed into the centres, so the rows here are those
        # within 0.8 of the cut, which leaves every row inside the cut the
        # model computes. Before the fix one more pass moved rows on OLD.
        moved = []
        for seed in range(seeds):
            x = eight_blobs(seed)
            x = x[((x - x.mean(0)) ** 2).sum(1) <= 0.8 * 5.0 * ((x - x.mean(0)) ** 2).sum(1).mean()]
            d2 = ((x - x.mean(0)) ** 2).sum(1)
            assert d2.max() < 0.95 * 5.0 * d2.mean(), "a row near the model's cut"
            spec = po.spec.kmeans(
                "m",
                features=["x0", "x1"],
                n_clusters=8,
                half_life=float("inf"),
                warm_rows=len(x),
                seed=seed,
                split_merge=0.0,
                standardize=False,
            )
            bank = po.ModelBank([spec])
            bank.fit(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]}))
            assert failures(bank) == 0
            c = bank.coef()["coef"].to_numpy().reshape(8, 2)
            label = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            means = np.array(
                [x[label == j].mean(0) if (label == j).any() else c[j] for j in range(8)]
            )
            again = ((x[:, None] - means[None]) ** 2).sum(-1).argmin(1)
            moved.append(int((again != label).sum()))
        print(f"the library: rows one more pass moves, by seed: {moved}")
        assert max(moved) == 0


class TestKnownDefects:
    """Caps that bind in silence (task 278 (b), 2026-10-25), pinned as they
    stand so that the fix flips the test rather than passing unnoticed."""

    def test_the_gram_lasso_path_runs_out_of_sweeps_in_silence(self):
        # `gram.lasso_path` runs the model's descent offline (`cd_path`) and
        # returns the coefficients whether or not a point met `tol`: no
        # count, no warning, no flag in its result. On a pair 1e-4 apart (a
        # condition of 4e8) at a penalty of 0.001, five sweeps and its
        # default 1,000 both leave the KKT violation at 7.8e-9 against the
        # rule's certificate of 2e-10 (measured 2026-10-25).
        x, y = collinear(800, 1, 1e-4)
        spec = po.spec.ewridge(
            "m", targets=["y"], features=FEATURES, ridge=0.0, half_life=float("inf")
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(frame(x, y))
        g = bank.gram("m")[0]
        c, d, s, _ = standardized_system(x, y)
        for max_iter in (5, 1000):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                path = gram.lasso_path(g, penalties=[0.001], max_iter=max_iter)
            b = np.asarray(path)[0, 1:] * s
            v = kkt_violation(c, d, 0.001, b)
            certified = 1e-10 * np.abs(c).sum(1)
            print(
                f"lasso_path, {max_iter} sweeps: KKT {v.max():.1e}, certified {certified.max():.1e}"
            )
            assert v.max() > 10 * certified.max(), "the descent met its rule after all"
            assert not caught, [str(w.message) for w in caught]
