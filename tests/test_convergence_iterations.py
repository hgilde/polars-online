"""Convergence class 8: inner-iteration honesty (docs/PLAN.md task 278).

Wherever this library iterates inside a row -- a coordinate descent, a
Lloyd iteration, a refit -- it stops by a rule or by a cap. The class holds
two things on inputs built to make the iteration slow:

- **the stopping rule was met** where the library says it converged, checked
  against what the rule certifies: the lasso's KKT conditions, and its
  solution next to scikit-learn's (taken as given, not tested);
- **every cap hit is counted** or said: in ``solve_failures``, or for
  ``gram.lasso_path`` in a ``ReadinessWarning``. Where one was silent, the
  silence was pinned with its measured rate until the fix flipped the test.

What iterates, and what each says about a cap hit (2026-10-25, at
``fe98f01``):

==========================================  ===========================  =========
the iteration                               its stopping rule and cap    a cap hit
==========================================  ===========================  =========
``lasso``'s descent, per target and point   a sweep's largest move at     counted in
                                            most ``tol`` times the        ``solve_failures``
                                            target's spread;
                                            ``max_iter`` sweeps
``gram.lasso_path``, the same descent       the same; 1,000 sweeps        a
                                                                          ``ReadinessWarning``
                                                                          naming the
                                                                          penalties
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
``huber``'s and ``sgd(loss="huber")``'s     no row's fit moves by more    counted in
refit of each target's first rows           than 1e-10 of the residual    ``solve_failures``
                                            scale; 200 IRLS passes
==========================================  ===========================  =========

Until task 278's fixes (D3, D6 and D7), the lasso's rule compared the move
with ``tol`` itself, in the target's units, ``gram.lasso_path``'s cap hits
were silent, and the ``lloyd`` seeding ran ten passes with no rule and no
count.

Until task 278's fix the refit's cap hits were silent, and its cap was 50
passes: over 3,000 streams of 50-500 rows, ``y = 1 + 2x + 0.5 e`` with a
level added to up to 45% of the rows with ``x > 0``, 50 passes ran out on
142, none below 20% of the rows; run on, every one met the rule, the
slowest after 880 passes. At 200 passes 6 run out, all past 39%
(``online_core``'s ``slow_refit_stream``; the Rust tests ``a_refit_that_runs_to_its_cap_is_counted``
and ``a_refit_that_runs_to_its_cap_is_a_solve_failure`` hold the count).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Lasso

import ledger
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


def lasso_case(
    eps: float, lam: float, max_iter: int, tol: float, n: int = 800, scale: float = 1.0
) -> dict[str, Any]:
    """One descent's honesty: was the last solve counted, and if not, is it
    within what its stopping rule certifies? `scale` multiplies the target
    and, as the penalty is in the target's units, the penalty."""
    x, y = collinear(n, 1, eps)
    y = y * scale
    lam = lam * scale
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
    # The stopping rule: in the last sweep every coordinate moved by at
    # most `tol * s_y`, `s_y` the target's spread (its population standard
    # deviation under `half_life = inf`). Coordinate `i` met its own
    # condition exactly when it was updated (the soft threshold is its
    # exact minimizer); the coordinates after it in the sweep then moved
    # its gradient by at most `tol * s_y` each, times `|C_ij|`. So the rule
    # certifies `|v_i| <= tol * s_y * sum_j |C_ij|`. The model's moments
    # and these, written from the rows, differ by the accumulation's
    # rounding, `gamma_{4n}` of the terms of `C b` and `d`.
    certified = tol * float(np.std(y)) * np.abs(c).sum(1)
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


#: The ledger's floor on a converged descent's KKT violation over its
#: certificate: twice the worst measured over `GRID` (2026-10-26), where the
#: violation is rounding.
LASSO_KKT_FLOOR = 0.0025

#: The oracle ledger's inputs (tests/ledger.py): the last solve's
#: coefficients and its count at every point of the grid at the defaults
#: (`max_iter` 100); the strict points run up to 100,000 sweeps a row.
ledger.guard(
    "lasso_descent",
    lambda: [
        lasso_case(eps, lam, max_iter, tol)["b"]
        for eps, lam, max_iter, tol in GRID
        if max_iter == 100
    ],
    bits=52,
)


class TestLassoDescent:
    """The descent stops when a sweep moves no standardized coefficient by
    more than `tol` times the target's spread, or after `max_iter` sweeps,
    which `solve_failures` counts. A descent not counted must be within what
    its rule certifies, and a descent that is not within it must have been
    counted.

    Measured 2026-10-25 over `GRID`, with the rule relative to the target's
    spread (1.1 on these streams): every uncounted last solve sat at or
    below 4.4e-14 of KKT violation (bounds 1e-11 to 2.3e-10), every counted
    one at 1.0e-11 to 1.2e-5; the defaults count a cap hit on 798 of 799
    solves at a penalty of 0.001 and a pair 0.1 apart (a condition of 4e2),
    and the strict settings still count every solve at 1e-4 (4e8): a cyclic
    descent there needs about `kappa` sweeps. The 798 are the descent's
    rate, not its rule: from a warm start about 1e-3 off, contracting by
    about 0.99 a sweep, it needs some 1,600 sweeps to meet 1e-10; at `tol`
    1e-6 it still counted 676, and at `max_iter` 1,000 it counted 349 at
    9.4 times the sweeps. A single penalty takes nothing from a warm start
    per path point.
    """

    @staticmethod
    def check(eps, lam, max_iter, tol, scale=1.0) -> dict[str, Any]:
        case = lasso_case(eps, lam, max_iter, tol, scale=scale)
        within = bool(np.all(case["kkt"] <= case["bound"]))
        print(
            f"eps {eps:g} l {lam:g} max_iter {max_iter} tol {tol:g} y x {scale:g}:"
            f" counted {case['counted']},"
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

    @pytest.mark.parametrize("power", [-30, 30])
    def test_a_descent_meets_its_rule_at_any_scale_of_the_target(self, power):
        # The rule is relative to the target's spread, so a descent on the
        # target scaled by 2^k (the penalty with it) is the unscaled one's,
        # scaled: it stops on the same sweep, counted or not, and within the
        # same certificate. When `tol` was compared with the move itself, in
        # the target's units, a target near 1e-9 met it in a sweep or two:
        # uncounted, its KKT violation 9.5e-19 against a certificate of 5.2e-21
        # (task 278, D3, measured 2026-10-25).
        base = self.check(1e-2, 0.01, 100_000, 1e-14)
        c = 2.0**power
        case = self.check(1e-2, 0.01, 100_000, 1e-14, scale=c)
        assert not case["counted"]
        assert case["counted"] == base["counted"]
        np.testing.assert_array_equal(case["b"], base["b"] * c)

    @pytest.mark.oracle_grid(reason="a grid: 32 descents and scikit-learn beside each")
    def test_every_descent_of_the_grid_meets_its_rule_or_is_counted(self):
        cases = [self.check(*point) for point in GRID]
        counted = [c["counted"] for c in cases]
        # A count is held exactly: one more descent at its cap is a
        # regression; a descent that stopped by its rule, by how far inside
        # its certificate it stopped.
        ledger.record("lasso_descent", "descents_at_the_cap", sum(counted), exact=True)
        within = [float(np.max(c["kkt"] / c["bound"])) for c in cases if not c["counted"]]
        ledger.record("lasso_descent", "kkt_over_certificate", within, floor=LASSO_KKT_FLOOR)
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


def lloyd_fit(seed: int) -> np.ndarray:
    """``kmeans``' centres, its default ``lloyd`` seeding, on one seed's
    eight overlapping blobs: the library's half of the Lloyd grid."""
    x = eight_blobs(seed)
    x = x[((x - x.mean(0)) ** 2).sum(1) <= 0.8 * 5.0 * ((x - x.mean(0)) ** 2).sum(1).mean()]
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
    return bank.coef()["coef"].to_numpy()


#: The oracle ledger's inputs: the library's seeding on ten of the grid's
#: fifty seeds (the reference is Python, a second each).
ledger.guard("kmeans_lloyd", lambda: [lloyd_fit(seed) for seed in range(10)])


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
                50,
                marks=pytest.mark.oracle_grid(reason="a grid: 50 seedings in the Python reference"),
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
        if seeds == 50:
            ledger.record("kmeans_lloyd", "reference_short_of_a_fixed_point", short, exact=True)
            ledger.record("kmeans_lloyd", "reference_runs_at_the_cap", hits, exact=True)
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
        if seeds == 50:
            ledger.record("kmeans_lloyd", "rows_one_more_pass_moves", moved, exact=True)
        assert max(moved) == 0


class TestGramLassoPath:
    """`gram.lasso_path` runs the model's descent offline, by the same rule,
    and says which penalties ran out of sweeps: a `ReadinessWarning` naming
    them, the coefficients returned as the descent left them. Until task
    278's fix (D6) it said nothing: on a pair 1e-4 apart (a condition of
    4e8) at a penalty of 0.001, five sweeps and its default 1,000 both left
    the KKT violation at 7.8e-9 against the rule's certificate of 2e-10
    (measured 2026-10-25)."""

    @staticmethod
    def path(eps: float, lam: float, **kw: Any) -> tuple[dict[str, Any], list[str]]:
        x, y = collinear(800, 1, eps)
        spec = po.spec.ewridge(
            "m", targets=["y"], features=FEATURES, ridge=0.0, half_life=float("inf")
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(frame(x, y))
        g = bank.gram("m")[0]
        c, d, s, _ = standardized_system(x, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = gram.lasso_path(g, penalties=[lam], **kw)
        b = np.asarray(path)[0, 1:] * s
        tol = kw.get("tol", 1e-10)
        # As `lasso_case`'s certificate: `tol * s_y * sum_j |C_ij|`, and the
        # rounding of the Gram's moments against these from the rows.
        bound = tol * float(np.std(y)) * np.abs(c).sum(1)
        bound += 8 * gamma(4 * len(y)) * (np.abs(c) @ np.abs(b) + np.abs(d) + lam)
        case = {"kkt": kkt_violation(c, d, lam, b), "bound": bound}
        return case, [str(w.message) for w in caught if issubclass(w.category, po.ReadinessWarning)]

    @pytest.mark.parametrize("max_iter", [5, 1000])
    def test_a_penalty_that_runs_out_of_sweeps_is_named_in_a_warning(self, max_iter):
        case, said = self.path(1e-4, 0.001, max_iter=max_iter)
        print(f"lasso_path, {max_iter} sweeps: KKT {case['kkt'].max():.1e}, said {said}")
        # The hard path ran: the descent is outside its certificate.
        assert case["kkt"].max() > 10 * case["bound"].max(), "the descent met its rule after all"
        assert len(said) == 1, said
        assert "1 of 1" in said[0] and "0.001" in said[0] and f"{max_iter} sweeps" in said[0]

    def test_a_path_that_meets_its_rule_says_nothing_and_is_within_it(self):
        case, said = self.path(1e-2, 0.01, max_iter=100_000, tol=1e-14)
        print(f"lasso_path, converged: KKT {case['kkt'].max():.1e} vs {case['bound'].max():.1e}")
        assert not said, said
        assert np.all(case["kkt"] <= case["bound"])


def slow_refit_stream(trial: int) -> pl.DataFrame:
    """``online_core``'s ``slow_refit_stream``, written out again: a xorshift
    stream whose length, share of moved rows and level are drawn from
    ``trial``. Trial 687 is 71 rows, 39% of the rows with ``x > 0`` moved,
    whose refit needs 880 IRLS passes."""
    mask = (1 << 64) - 1
    state = ((trial + 1) * 0x9E3779B97F4A7C15) & mask

    def rnd() -> float:
        nonlocal state
        state ^= (state << 13) & mask
        state ^= state >> 7
        state ^= (state << 17) & mask
        return (state >> 11) / float(1 << 53)

    n = 50 + int(rnd() * 450.0)
    frac = rnd() * 0.45
    level = math.exp(rnd() * 6.0 - 1.0) * 3.0
    xs, ys = [], []
    for _ in range(n):
        x = rnd() * 2.0 - 1.0
        e = math.sqrt(-2.0 * math.log(max(rnd(), 1e-300))) * math.cos(math.tau * rnd())
        out = level if rnd() < frac and x > 0.0 else 0.0
        xs.append(x)
        ys.append(1.0 + 2.0 * x + 0.5 * e + out)
    return pl.DataFrame({"x0": xs, "y": ys})


class TestHuberRefit:
    """The refit of each target's first rows, ``huber``'s and
    ``sgd(loss="huber")``'s: IRLS until no row's fit moves by more than
    1e-10 of the residual scale, at most 200 passes, and a refit that takes all
    200 counted in ``solve_failures`` (task 278). Trial 687's stream runs to
    the cap; trial 0's converges inside it; the refit off counts nothing."""

    @pytest.mark.parametrize("kind", ["huber", "sgd"])
    def test_a_refit_at_its_cap_is_counted(self, kind):
        extra = {"loss": "huber", "learning_rate": 0.05} if kind == "sgd" else {}
        for trial, rows, want in [(687, None, 1), (0, None, 0), (687, 0, 0)]:
            df = slow_refit_stream(trial)
            spec = getattr(po.spec, kind)(
                "m",
                targets=["y"],
                features=["x0"],
                half_life=float("inf"),
                min_weight=3.0,
                huber_refit_rows=df.height if rows is None else rows,
                **extra,
            )
            bank = po.ModelBank([spec])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bank.fit_predict(df)
            print(f"{kind}, trial {trial}, refit rows {rows}: {failures(bank)}")
            assert failures(bank) == want, (kind, trial, rows)
