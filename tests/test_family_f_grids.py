"""Worst-seed grids for the clustering and changepoint models' own keywords
(task 278, batch 33; PARAMS family F, F6).

Each relation below was held, before this module, by one stream at one
seed. Here each is derived where it can be, measured where it cannot, and
judged by the worst seed and the 95th percentile, never a median:

- ``kmeans``: a centre whose blob vanished is re-placed by the dead rule
  ``log2(1 / dead_frac)`` half-lives later, at the first split-merge check
  after its weight crosses the floor; the delay grows with the checks'
  spacing; and a re-placed centre is born at half its source's weight, so a
  ``dead_frac`` of 0.5 or more re-places it again at once (a defect, pinned
  by strict xfails).
- ``corrchange`` (``kind="window"``): the permutation test's size on spans
  that do not overlap is the exact permutation level ``(n_perm + 1 - q) /
  (n_perm + 1)``, ``q = ceil((1 - alpha) n_perm)``, over 200 seeds.
- ``bocpd``: ``robust_beta`` trades a forgiven outlier against a detected
  shift, over 50 seeds.
- ``hmm``: the learned transition matrix recovers the truth at every
  ``transition_prior`` above 0, its error falling with the rows, and the
  prior pulls it toward the prior mean, monotonically; at 0 a state can be
  locked out (the documented limit), over 50 seeds.

The grids are weekly (``pytest.mark.soak``, run by the leak-check
workflow's ``pytest -m soak``); one cheap case of each runs in the
essentials. The module's ``TIER`` counts only ``extended`` and
``oracle_grid`` marks, so a module with soak marks on some tests and none on
the rest is ``essential``. Every model is held to the truth it was generated
from or to a relation derived from its own definition: nothing here scores
a third-party library.
"""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import pytest

import polars_online as po
from test_bocpd import normals
from test_bocpd import run as bocpd_run
from test_corrchange import pair
from test_kmeans import _state, ari, frame, spec, stranded, unnested

TIER = "essential"


def tail(values) -> tuple[float, float]:
    """The worst and the 95th percentile of ``values``."""
    v = np.asarray(values, dtype=float)
    return float(v.max()), float(np.quantile(v, 0.95))


# --- kmeans: the dead rule ----------------------------------------------------
#
# `stranded` (tests/test_kmeans.py): four blobs on a circle; at n/2 blob 3
# dies and a fifth is born at (9, 9), far from every centre, so its rows are
# far rows of the centres they fall nearest to and are learned into none.
# The stranded centre (blob 3's) then learns no row, and its weight decays
# by lam = 2^(-1/h) a row, while the floor `dead_frac * W / k` stays put (W
# is the shared weight, within 0.1% of steady at n/2, ten half-lives in).
# The weight crosses the floor after
#
#     r = h log2(n_j / (dead_frac W / k)) = h log2(1 / dead_frac) + h log2(share)
#
# rows, `share` = n_j / (W / k), 1 up to the decayed count's binomial noise:
# sd sqrt(0.75 / 4 * h / (2 ln 2)) / (h / (4 ln 2)) = 3.2% at h = 1000, 0.046
# half-lives in log2. The rule fires at the first split-merge check at or
# after r: checks fall every `every` learned rows, so the check is q with
# 0 <= q - r < every, and q is predicted to the row from the state at n/2
# (n_j, W and the check counter `since_sm`). The re-placed centre takes the
# far rows' mean, blob 5's, so the first blob-5 row after q is labelled with
# the stranded centre: a lag geometric in 1/4, P(lag > 40) = 0.75^41 = 7e-6
# a case.

KM_N, KM_H = 20_000, 1000.0
KM_SEEDS = range(50)
#: The lag from the predicted check to the first blob-5 row it labels, a
#: geometric variable in 1/4 (above): 7e-6 a case past 40. Measured
#: 2026-10-26 over 50 seeds x 12 cells: worst 19, p95 10.
KM_LAG = 40
#: `log2(share)`'s spread in half-lives, 4.3 of its sd 0.046 (above), the
#: band around `log2(1 / dead_frac)` beside the checks' spacing and the lag.
#: Measured 2026-10-26: the low edge -0.099 (seed and cell below).
KM_SHARE = 0.2


def replacement(seed: int, dead_frac: float, every: int) -> dict:
    """One `stranded` stream: where the dead rule re-placed the stranded
    centre, against the check the state at n/2 predicts."""
    X, lab = stranded(seed=seed, n=KM_N)
    half = KM_N // 2
    bank = po.ModelBank(
        [
            spec(
                k=4,
                half_life=KM_H,
                min_weight=1.0,
                warm_rows=200,
                split_merge=0.5,
                split_merge_every_rows=every,
                dead_frac=dead_frac,
            )
        ]
    )
    first = unnested(bank.fit_predict(frame(X[:half])))["cluster"].fill_null(-1).to_numpy()
    at_death = _state(bank)
    second = unnested(bank.fit_predict(frame(X[half:])))["cluster"].fill_null(-1).to_numpy()
    end = _state(bank)
    own = first[lab[:half] == 3][-500:]
    j = int(np.bincount(own[own >= 0]).argmax())
    n_j = at_death["clusters"][j]["n"]
    # Row by row from n/2: the centre's weight decays, the shared weight
    # decays and takes the row (it is still 0.1% short of its steady value
    # at n/2, so the floor rises by a row or two's worth over the delay).
    lam = 0.5 ** (1.0 / KM_H)
    w = at_death["moments"]["w"]
    crossing, row = None, 0
    while True:
        row += 1
        n_j *= lam
        w = lam * w + 1.0
        if crossing is None and n_j < dead_frac * w / 4:
            crossing = row
        if crossing is not None and (at_death["since_sm"] + row) % every == 0:
            check = row
            break
    hit = np.flatnonzero((lab[half:] == 4) & (second == j))
    late = np.arange(KM_N) >= KM_N - KM_N // 4
    got = np.concatenate([first, second])
    return {
        "replaced": int(hit[0]) if len(hit) else None,
        "check": check,
        "crossing": crossing,
        "n_dead": end["n_dead"],
        "tail_ari": ari(lab[late], got[late]),
    }


def km_holds(r: dict, every: int) -> None:
    assert r["n_dead"] == 1, r
    assert r["replaced"] is not None, r
    assert 0 <= r["check"] - r["crossing"] < every, r
    assert 0 <= r["replaced"] - r["check"] <= KM_LAG, r
    assert r["tail_ari"] > 0.95, r


def test_one_stranded_centre_is_re_placed_at_the_check_its_weight_predicts():
    """The essentials' case: one seed at the default ``dead_frac``. Measured
    2026-10-26: the check the state predicts at 4,400 rows after the blob
    died, the first row labelled so 13 rows later, 4.41 half-lives against
    ``log2(20)`` = 4.32."""
    r = replacement(1, 0.05, 100)
    print(f"seed 1: {r}")
    km_holds(r, 100)
    e = r["replaced"] / KM_H - math.log2(1 / 0.05)
    assert -KM_SHARE <= e <= 100 / KM_H + KM_SHARE + KM_LAG / KM_H, e


#: A re-placed centre is born at half its source's weight (`n_l /= 2`), and
#: the source, holding about an equal share `W / k`, keeps the other half:
#: both sit at `W / (2k)`, which is the floor at `dead_frac = 0.5`. From
#: there on whichever is lighter at the next check is "dead" and goes to
#: whatever far rows that check has seen, the blobs' own chi-square tails
#: included, and the clusters are lost. Measured 2026-10-26 over 50 seeds at
#: a check every 25 rows: no cascade at 0.3, 0.4 or 0.45; at 0.5, 3 seeds
#: (n_dead up to 210, late ARI 0.50); at 0.6, 29 (ARI 0.23); at 0.9, all 50
#: (n_dead up to 518). `dead_frac` is refused from 1, not from 0.5.
CASCADE = (
    "defect: a centre re-placed by the dead rule is born at half its source's "
    "weight, at or below the floor once dead_frac >= 0.5, and the rule re-places "
    "again at once; dead_frac is refused only from 1 (task 278 batch 33)"
)

KM_CELLS = [
    pytest.param(
        d,
        e,
        marks=[pytest.mark.xfail(strict=True, reason=CASCADE)]
        if (d, e) in {(0.5, 25), (0.5, 100)}
        else [],
        id=f"{d}-{e}",
    )
    for d in (0.05, 0.1, 0.25, 0.5)
    for e in (25, 100, 400)
]


@pytest.mark.soak
@pytest.mark.parametrize("dead_frac, every", KM_CELLS)
def test_a_vanished_blob_s_centre_is_re_placed_log2_one_over_dead_frac_half_lives_later(
    dead_frac, every
):
    """Every seed: one re-placement, at the check the state predicts, the
    late quarter clustered again; the delay within the band around
    ``log2(1 / dead_frac)`` half-lives that the share's noise, the spacing
    and the lag allow. Measured 2026-10-26 (``e`` = delay in half-lives less
    ``log2(1 / dead_frac)``): ``e`` from -0.099 to 0.295 at 0.05 (the high
    end a check every 400 rows), to 0.089 at 0.1, 0.212 at 0.25, 0.408 at
    0.5; lag worst 19. At 0.5 the checks every 25 and 100 rows cascade
    (``CASCADE``)."""
    rs = [replacement(s, dead_frac, every) for s in KM_SEEDS]
    found = [r for r in rs if r["replaced"] is not None]
    e = [r["replaced"] / KM_H - math.log2(1 / dead_frac) for r in found]
    lag = [r["replaced"] - r["check"] for r in found]
    print(
        f"dead_frac {dead_frac}, every {every}: n_dead {sorted({r['n_dead'] for r in rs})}; "
        f"e from {min(e):.3f} to {max(e):.3f}, |e| worst/p95 {tail(np.abs(e))}; "
        f"lag worst/p95 {tail(lag)} (seed {int(np.argmax(lag))})"
    )
    for r in rs:
        km_holds(r, every)
    assert min(e) >= -KM_SHARE, e
    assert max(e) <= every / KM_H + KM_SHARE + KM_LAG / KM_H, e


@pytest.mark.soak
def test_the_delay_grows_with_the_check_spacing():
    """Checks every 25, 50, 100, 200 and 400 rows fall on nested rows (the
    counter starts at seeding, the same row in all five), so for each seed
    the check that re-places the centre, and the first blob-5 row after it,
    can only come later as the spacing grows; each within one spacing of
    the crossing. At ``dead_frac = 0.25``, clear of the cascade. Measured
    2026-10-26: the check's excess over the crossing averaged 12, 25, 55, 109
    and 209 rows, about half a spacing; worst 24, 49, 99, 199 and 310."""
    spacings = (25, 50, 100, 200, 400)
    excess = {e: [] for e in spacings}
    for s in KM_SEEDS:
        rs = {e: replacement(s, 0.25, e) for e in spacings}
        for e, r in rs.items():
            km_holds(r, e)
            excess[e].append(r["check"] - r["crossing"])
        replaced = [rs[e]["replaced"] for e in spacings]
        assert replaced == sorted(replaced), (s, replaced)
    means = [float(np.mean(excess[e])) for e in spacings]
    by = dict(zip(spacings, means, strict=True))
    print(f"excess of the check over the crossing, mean by spacing {by}")
    print(f"worst/p95 by spacing {[tail(excess[e]) for e in spacings]}")
    assert means == sorted(means), means
    assert means[-1] > 4 * means[0], means


@pytest.mark.xfail(strict=True, reason=CASCADE)
def test_a_re_placed_centre_is_not_born_dead_one_seed():
    """The essentials' repro of ``CASCADE``: seed 0 at ``dead_frac = 0.6``,
    a check every 25 rows. Measured 2026-10-26: re-placed again and again."""
    r = replacement(0, 0.6, 25)
    print(f"seed 0, dead_frac 0.6: {r}")
    km_holds(r, 25)


@pytest.mark.soak
@pytest.mark.xfail(strict=True, reason=CASCADE)
@pytest.mark.parametrize("dead_frac", [0.5, 0.6, 0.9])
def test_a_re_placed_centre_is_not_born_below_the_floor(dead_frac):
    """What a fix must make true: one re-placement and the late quarter
    clustered on every seed, for any ``dead_frac`` the builder accepts."""
    rs = [replacement(s, dead_frac, 25) for s in KM_SEEDS]
    bad = [
        (s, r["n_dead"], round(r["tail_ari"], 3))
        for s, r in zip(KM_SEEDS, rs, strict=True)
        if r["n_dead"] > 1
    ]
    print(f"dead_frac {dead_frac}: {len(bad)} seeds cascade {bad[:5]}")
    assert not bad, bad
    for r in rs:
        km_holds(r, 25)


@pytest.mark.soak
@pytest.mark.parametrize("dead_frac", [0.3, 0.45])
@pytest.mark.parametrize("every", [10, 25])
def test_below_half_a_share_a_centre_is_re_placed_once(dead_frac, every):
    """The other side of ``CASCADE``'s edge: below 0.5 a re-placed centre
    and its halved source are born above the floor, and every seed sees one
    re-placement however close the checks. Measured 2026-10-26 at 0.3, 0.4
    and 0.45, checks every 10 and 25 rows: one on every seed."""
    rs = [replacement(s, dead_frac, every) for s in KM_SEEDS]
    print(f"dead_frac {dead_frac}, every {every}: n_dead {sorted({r['n_dead'] for r in rs})}")
    for r in rs:
        km_holds(r, every)


# --- corrchange: the window kind's permutation size ---------------------------
#
# The critical value is the q-th smallest of n_perm statistics of the pooled
# rows shuffled between the two windows, q = ceil((1 - alpha) n_perm), and a
# report flags when its statistic is above it. On rows that are exchangeable
# (iid, one correlation throughout) the observed split is one more draw of
# the same distribution, so it exceeds the q-th smallest of n_perm draws with
# probability exactly (n_perm + 1 - q) / (n_perm + 1): the permutation test's
# level, which is alpha only up to that rounding (0.0952 at alpha 0.05 and
# 20 draws; 0.0476 at alpha 0.01 and 20 draws). Blocks of rows permute an
# iid stream exchangeably too. Two windows that slide by a row share all but
# one row, so the trials are the reports on spans that do not overlap: with
# `permute_every_rows = 2 w` a fresh value is drawn, from the previous row's
# windows, for each report at rows 2w + 2w j, which are those spans. A seed's
# flag count over its J spans is binomial in that level, so the worst of 200
# seeds is held to the binomial's 1 - 0.001/200 quantile and the pooled rate
# to 4 standard errors.

CC_W, CC_K, CC_SPANS = 50, 3, 40


def binomial_upper(n: int, p: float, tail_mass: float) -> int:
    """The smallest ``c`` with ``P(X > c) <= tail_mass`` for ``X`` binomial."""
    cdf = 0.0
    for c in range(n + 1):
        cdf += math.comb(n, c) * p**c * (1 - p) ** (n - c)
        if 1.0 - cdf <= tail_mass:
            return c
    return n


def level(alpha: float, n_perm: int) -> float:
    """The permutation test's exact level (above)."""
    q = math.ceil((1.0 - alpha) * n_perm)
    return (n_perm + 1 - q) / (n_perm + 1)


def null_spans(seeds: range) -> pl.DataFrame:
    n = 2 * CC_W * (CC_SPANS + 1)
    return pl.concat(
        [pair(n, 0.4, seed=10_000 + s, k=CC_K).with_columns(g=pl.lit(s)) for s in seeds]
    )


def span_flags(seeds: range, alphas, n_perm: int, perm_block: int) -> dict[float, np.ndarray]:
    """Each seed's flag count over its spans at each level, after checking
    that every sampled report has a freshly drawn critical value."""
    df = null_spans(seeds)
    cols = [f"x{i}" for i in range(CC_K)]
    specs = [
        po.spec.corrchange(
            f"a{i}",
            features=cols,
            kind="window",
            window_rows=CC_W,
            alpha=a,
            n_perm=n_perm,
            perm_block=perm_block,
            permute_every_rows=2 * CC_W,
            group="g",
        )
        for i, a in enumerate(alphas)
    ]
    out = po.ModelBank(specs).fit_predict(df)
    n = 2 * CC_W * (CC_SPANS + 1)
    rows = 2 * CC_W + 2 * CC_W * np.arange(CC_SPANS)
    counts = {}
    for i, a in enumerate(alphas):
        o = out[f"a{i}"].struct.unnest()
        crit = o["crit"].to_numpy().reshape(len(seeds), n)
        flag = o["flag"].fill_null(False).to_numpy().reshape(len(seeds), n)
        assert np.isfinite(crit[:, rows]).all()
        # Evidence each sampled report reads a value drawn for it.
        assert (crit[:, rows] != crit[:, rows - 1]).all()
        counts[a] = flag[:, rows].sum(axis=1)
    return counts


def cc_holds(counts: np.ndarray, alpha: float, n_perm: int, seeds: int) -> str:
    want = level(alpha, n_perm)
    trials = seeds * CC_SPANS
    pooled = counts.sum() / trials
    se = math.sqrt(want * (1 - want) / trials)
    worst, p95 = tail(counts)
    bound = binomial_upper(CC_SPANS, want, 0.001 / seeds)
    line = (
        f"alpha {alpha}, n_perm {n_perm}: level {want:.4f}, pooled {pooled:.4f} "
        f"({(pooled - want) / se:+.2f} se); a seed's flags worst {worst:.0f}/{CC_SPANS} "
        f"(bound {bound}), p95 {p95:.1f}"
    )
    assert abs(pooled - want) < 4 * se, line
    assert worst <= bound, line
    return line


def test_the_window_permutation_size_is_its_exact_level_twenty_seeds():
    """The essentials' case: 20 seeds of 40 spans at the smallest ``n_perm``,
    whose level at 0.05 is 2/21 = 0.0952. Measured 2026-10-26: 0.0900
    pooled over these 20 seeds (-0.5 se), 0.1006 over 200 (+1.6 se)."""
    counts = span_flags(range(20), (0.05,), 20, 1)[0.05]
    print(cc_holds(counts, 0.05, 20, 20))


@pytest.mark.soak
@pytest.mark.parametrize("n_perm, perm_block", [(20, 1), (200, 1), (200, 5)])
def test_the_window_permutation_size_is_its_exact_level(n_perm, perm_block):
    """200 seeds of 40 spans each, at three levels. Measured 2026-10-26 at
    200 draws in blocks of 1: pooled 0.0150 / 0.0551 / 0.2042 against the
    levels 0.0149 / 0.0547 / 0.2040 (within 0.2 standard errors), a seed's
    worst 3 / 8 / 16 flags of 40 against the bounds 6 / 11 / 21; in blocks
    of 5, pooled 0.0149 / 0.0583 / 0.2061 (+1.4 se at most), worst 3 / 8 /
    17; at 20 draws, levels 0.0476 / 0.0952 / 0.2381 -- nearly five times
    ``alpha`` at 0.01 -- pooled 0.0475 / 0.1006 / 0.2424 (+1.6 se at most),
    worst 6 / 9 / 17 against 10 / 14 / 23."""
    seeds = range(200)
    alphas = (0.01, 0.05, 0.2)
    counts = span_flags(seeds, alphas, n_perm, perm_block)
    for a in alphas:
        print(f"block {perm_block}: " + cc_holds(counts[a], a, n_perm, len(seeds)))


# --- bocpd: robust_beta's trade -------------------------------------------------
#
# Tempered, each row is weighted by (pi(x) / pi(mode)) ** beta, so a row far
# out under every run is about equally unlikely under all of them and moves
# none: a 20-sigma outlier leaves p_change = P(r <= 1) near its floor, the
# mass at r = 0, which is exactly H = 1 / hazard on every row. A new regime is
# a run of such rows, each forgiven, so the same knob delays and then hides a
# real shift. Untempered, the outlier is a changepoint and the shift is found
# at once. Detection is read from the run mode: the first row after the
# shift whose run began within 5 rows of it.

BP_SEEDS = range(50)
BP_KW = dict(hazard=250.0, mean_prior=[0.0], covariance_prior=[1.0], degrees_of_freedom_prior=2.0)
BP_H = 1.0 / 250.0


def shift_found(seed: int, **kw) -> int | None:
    """Rows from a four-sigma shift at row 150 to the first row whose run
    began within 5 rows of it, or None within the 150 rows after."""
    x = np.concatenate([normals(150, 2 * seed + 1000), normals(150, 2 * seed + 1001, loc=4.0)])
    mode = bocpd_run(x, **BP_KW, **kw)["run_mode"].to_numpy()
    t = np.arange(len(x))
    ok = np.flatnonzero((t >= 150) & (np.abs(t - mode - 150) <= 5))
    return int(ok[0]) - 150 if len(ok) else None


def outlier_alarm(seed: int, **kw) -> float:
    """``p_change`` on a 20-sigma row in an otherwise quiet stream."""
    y = normals(200, seed + 5000)
    y[100] = 20.0
    return bocpd_run(y, **BP_KW, **kw)["p_change"][100]


def test_a_small_robust_beta_forgives_the_outlier_and_finds_the_shift_one_seed():
    """The essentials' case at 0.1: the outlier leaves ``p_change`` within
    ``[H, 2H]`` where the untempered posterior calls it a change, and the
    shift is found within 10 rows."""
    plain, tempered = outlier_alarm(0), outlier_alarm(0, robust_beta=0.1)
    found = shift_found(0, robust_beta=0.1)
    print(f"seed 0: p_change plain {plain:.4f}, tempered {tempered:.5f}; found after {found} rows")
    assert plain > 0.3
    assert BP_H * (1 - 1e-9) <= tempered <= 2 * BP_H
    assert found is not None and found <= 10


@pytest.mark.soak
def test_robust_beta_trades_the_outlier_against_the_shift():
    """Over 50 seeds (measured 2026-10-26; delay in rows, worst / p95):

    ========  ===========  ============  ======================
    beta      found        delay         outlier ``p_change``
    ========  ===========  ============  ======================
    none      50           3 / 3         0.34 at least
    0.05      50           4 / 3         0.0041 at most
    0.1       50           6 / 5         0.0041
    0.15      50           16 / 14.5     0.0043
    0.2       38           120 / 59.8    0.0047
    0.3       1            59            0.0058
    0.5       0                          0.0074
    ========  ===========  ============  ======================

    So the docstring's edge, "above about 0.2 nothing is ever detected
    again", is a soft one: at 0.2 three streams in four still date the
    shift, late, and at 0.25 three in fifty. Held: every seed found within
    10 rows up to 0.1; some seed missed at 0.2; at most 3 found at 0.3 and
    none at 0.5; the count of streams found never rises with beta, nor the
    worst delay falls while every stream is found; and from 0.05 up the
    outlier leaves ``p_change`` within ``[H, 2H]`` on every seed (``H`` is
    exact: the mass at ``r = 0``), where the untempered posterior reads 0.3
    or more on every seed."""
    betas = (None, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
    found, alarm = {}, {}
    for b in betas:
        kw = {} if b is None else {"robust_beta": b}
        found[b] = [shift_found(s, **kw) for s in BP_SEEDS]
        alarm[b] = [outlier_alarm(s, **kw) for s in BP_SEEDS]
        got = [d for d in found[b] if d is not None]
        worst = tail(got) if got else None
        print(
            f"beta {b}: found {len(got)}/50, delay worst/p95 {worst}; "
            f"outlier p_change {min(alarm[b]):.4f}..{max(alarm[b]):.4f}"
        )
    assert min(alarm[None]) > 0.3, "the untempered posterior calls the outlier a change"
    for b in betas[1:]:
        assert min(alarm[b]) >= BP_H * (1 - 1e-9) and max(alarm[b]) <= 2 * BP_H, b
    n_found = [sum(d is not None for d in found[b]) for b in betas]
    assert n_found == sorted(n_found, reverse=True), n_found
    for b in (None, 0.05, 0.1):
        assert None not in found[b] and max(found[b]) <= 10, (b, found[b])
    full = [b for b in betas if None not in found[b]]
    worst = [max(found[b]) for b in full]
    assert worst == sorted(worst), (full, worst)
    assert 0 < n_found[betas.index(0.2)] < 50, n_found
    assert n_found[betas.index(0.3)] <= 3, n_found
    assert n_found[betas.index(0.5)] == 0, n_found


# --- hmm: transition_prior and the learned matrix ---------------------------------
#
# A sticky two-state chain, the states three noise sds apart, given at their
# true means and covariances and learned from there. With no `transition`
# the prior mean is flat, so the matrix a row reads is
#
#     Pi_kl = (N_kl + tau) / (sum_l N_kl + K tau)
#
# from the learned counts N (the state's `a`, row-major). Under the Dirichlet
# arithmetic alone, tau's pull toward 1/K would be tau K |1/K - P| / (N_k +
# tau K): 0.006 at tau = 10 and 4,000 rows. The learned matrix moves more
# than that -- 0.088 at the 95th percentile -- because the counts are the
# filtered joint of consecutive states, read through the matrix being learned:
# a less sticky matrix makes the filter credit more switches, which makes the
# matrix less sticky still. The pull falls with the rows all the same (the
# rate test). At tau = 0 a state the first rows did not favour can get no
# count and stay out (the docstring's limit), here at only three sds.

HM_PT = np.array([[0.95, 0.05], [0.10, 0.90]])
HM_MU = [-1.5, 1.5]
HM_SEEDS = range(50)


def chain(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = np.empty(n, dtype=int)
    z[0] = rng.integers(2)
    u = rng.random(n)
    for t in range(1, n):
        z[t] = int(u[t] < HM_PT[z[t - 1], 1])
    return np.asarray(HM_MU)[z] + rng.standard_normal(n)


def hmm_bank(tau: float) -> po.ModelBank:
    s = po.spec.hmm(
        "m",
        features=["x"],
        n_components=2,
        precision_prior=1e-2,
        half_life=float("inf"),
        transition_prior=tau,
        means=HM_MU,
        covs=[1.0, 1.0],
    )
    return po.ModelBank([s])


def counts(bank: po.ModelBank) -> np.ndarray:
    model = json.loads(bank.write_json())["states"][0][0][1]["models"][0]["model"]
    return np.array(model["Hmm"]["a"]).reshape(2, 2)


def learned(x: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """The matrix the next row would read, and the counts it is read from."""
    bank = hmm_bank(tau)
    bank.fit_predict(pl.DataFrame({"x": x}))
    a = counts(bank)
    return (a + tau) / (a.sum(axis=1, keepdims=True) + 2 * tau), a


def test_the_learned_matrix_recovers_the_chain_one_seed():
    """The essentials' case: the formula above is the matrix the filter
    reads (the predicted state of the next row from its filtered one, to
    1e-15), and at the default prior one seed's matrix is within 0.08 of the
    truth (measured 2026-10-26: 0.012 on seed 0)."""
    x = chain(3, 601)
    bank = hmm_bank(0.3)
    bank.fit_predict(pl.DataFrame({"x": x[:-1]}))
    a = counts(bank)
    pi = (a + 0.3) / (a.sum(axis=1, keepdims=True) + 0.6)
    o = bank.fit_predict(pl.DataFrame({"x": x[-1:]}))["m"].struct.unnest()
    p = np.array([o["filtered_state0"][0], o["filtered_state1"][0]])
    pred = np.array([o["predicted_state0"][0], o["predicted_state1"][0]])
    assert np.abs(p @ pi - pred).max() <= 1e-15
    est, _ = learned(chain(0, 4000), 1.0)
    err = float(np.abs(est - HM_PT).max())
    print(f"seed 0, tau 1: {est.ravel()} against {HM_PT.ravel()}, error {err:.4f}")
    assert err < 0.08


#: The worst seed's largest entry error at 4,000 rows, by `transition_prior`;
#: measured 2026-10-26, worst / p95: 0.1 -> 0.074 / 0.035, 1 -> 0.055 /
#: 0.035, 10 -> 0.112 / 0.088, 100 -> 0.237 / 0.201 (the pull above). Half
#: as much again for headroom; the pull's full size is 0.45, the matrix flat.
HM_BOUND = {0.1: 0.11, 1.0: 0.08, 10.0: 0.17, 100.0: 0.33}


@pytest.mark.soak
def test_transition_prior_recovers_the_chain_and_pulls_it_toward_the_prior_mean():
    """Over 50 seeds of 4,000 rows: at each ``transition_prior`` above 0 the
    worst seed is within ``HM_BOUND``; from 1 up the matrix's distance from
    the prior mean falls on every seed as the prior grows, to within
    ``n / (4 tau)`` of it at ``1e6`` (the counts weigh ``n`` at most against
    ``2 tau``); and every state keeps a share of the counts. At 0, a state is
    locked out on some seeds: under 10% of the counts where the chain spends
    a third of its rows in it. Measured 2026-10-26: 7 seeds of 50 locked out
    at 0 (seed 12: 1.8%), none at 0.1 or above (least share 0.283). Between
    0.1 and 1 the distance rose on 5 seeds, by up to 0.059: below a
    pseudo-count of one the early rows' path, not the prior, sets the
    matrix."""
    taus = (0.0, 0.1, 1.0, 10.0, 100.0, 1e6)
    err = np.zeros((len(HM_SEEDS), len(taus)))
    far = np.zeros_like(err)
    share = np.zeros_like(err)
    for i, s in enumerate(HM_SEEDS):
        x = chain(s, 4000)
        for j, tau in enumerate(taus):
            est, a = learned(x, tau)
            err[i, j] = np.abs(est - HM_PT).max()
            far[i, j] = np.abs(est - 0.5).max()
            share[i, j] = a.sum(axis=1).min() / a.sum()
    for j, tau in enumerate(taus):
        worst, p95 = tail(err[:, j])
        print(
            f"tau {tau:g}: error worst {worst:.4f} (seed {int(err[:, j].argmax())}), "
            f"p95 {p95:.4f}; "
            f"least state share {share[:, j].min():.3f}, locked out {(share[:, j] < 0.1).sum()}"
        )
    for j, tau in enumerate(taus):
        if tau in HM_BOUND:
            assert err[:, j].max() < HM_BOUND[tau], tau
    up = [taus.index(t) for t in (1.0, 10.0, 100.0, 1e6)]
    assert (np.diff(far[:, up], axis=1) <= 0).all()
    assert far[:, taus.index(1e6)].max() <= 4000 / (4 * 1e6)
    assert (share[:, 1:] > 0.25).all()
    assert (share[:, 0] < 0.1).any(), "the documented limit at transition_prior = 0 no longer shows"


@pytest.mark.soak
def test_the_learned_matrix_converges_with_the_rows():
    """At the default prior the error falls like an estimate's: over 50 seeds
    at 4,000, 16,000 and 64,000 rows the worst seed falls at each step, the
    95th percentile with a log-log slope of -0.35 or steeper, and 64,000
    rows' worst is under 0.012. Measured 2026-10-26, worst / p95: 0.0548 /
    0.0346, 0.0184 / 0.0161, 0.0078 / 0.0066 (slope -0.60)."""
    sizes = (4000, 16_000, 64_000)
    worst, p95 = [], []
    for n in sizes:
        e = [float(np.abs(learned(chain(s, n), 1.0)[0] - HM_PT).max()) for s in HM_SEEDS]
        w, p = tail(e)
        worst.append(w)
        p95.append(p)
        print(f"{n} rows: worst {w:.4f} (seed {int(np.argmax(e))}), p95 {p:.4f}")
    assert worst == sorted(worst, reverse=True) and len(set(worst)) == 3, worst
    assert math.log(p95[-1] / p95[0]) / math.log(sizes[-1] / sizes[0]) < -0.35, p95
    assert worst[-1] < 0.012, worst
