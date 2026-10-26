"""The refit of `marginal(bin_stat="huber")`'s first rows, `bin_refit_rows`
(task 278): once its count of learned rows is reached, every target's
location and robust scale and every cell's location are refit to the
converged batch fit of the rows so far, and the stream goes on online.

One pass weighs each row against the location and the scale before it. Over
the first rows those are still settling, so a share of spikes pulls the
first locations, the scale learns the pulled deviations, the cut is drawn
wide, and without decay the bins stay where the spikes put them: captured,
as one pass of `huber` is.

The oracle is statsmodels' `RLM` with `HuberT(1.345)`: the target's joint
M-location and MAD scale (`scale_est="mad"`), then each bin's M-location at
that scale held fixed (`start_scale=s, update_scale=False`), over the rows
fed so far.
"""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT

import polars_online as po
from polars_online import _polars_online as native
from polars_online import _spec

TIER = "mixed"

DELTA = 1.345
INF = float("inf")
EDGES = [float(e) for e in np.linspace(-1.0, 1.0, 17)[1:-1]]


def stream(n: int, frac: float, seed: int) -> pl.DataFrame:
    """`tests/test_marginal_bin_huber.py`'s curve: x ~ U(-1, 1), y = 2 sin 3x
    + 0.5 eps, and +50 (about 20 of the target's scales) on a share `frac`
    of the rows."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = 2.0 * np.sin(3.0 * x) + 0.5 * rng.standard_normal(n)
    y = y + np.where(rng.random(n) < frac, 50.0, 0.0)
    return pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y})


def spec(**kw) -> dict:
    d = dict(
        targets=["y"],
        features=["x"],
        clock="t",
        half_life=INF,
        gap_cap=1e12,
        min_weight=2.0,
        bin_edges=[EDGES],
        bin_stat="huber",
    )
    d.update(kw)
    if d.get("bin_edges") is None:
        d.pop("bin_edges")
    return po.spec.marginal("m", **d)


def locations(df: pl.DataFrame, cuts: list[int], **kw) -> dict[int, tuple[np.ndarray, list]]:
    """`bin_mean_y` and `bin_edges` after each row count in `cuts`."""
    bank = po.ModelBank([spec(**kw)])
    out, start = {}, 0
    for cut in cuts:
        bank.fit_predict(df.slice(start, cut - start))
        start = cut
        r = bank.marginal("m").row(0, named=True)
        out[cut] = (np.asarray(r["bin_mean_y"], dtype=float), r["bin_edges"])
    return out


def oracle(y: np.ndarray, x: np.ndarray, edges: list) -> tuple[np.ndarray, float]:
    """statsmodels: each bin's M-location at the target's joint MAD scale."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sm.RLM(y, np.ones((len(y), 1)), M=HuberT(DELTA)).fit(
            scale_est="mad", conv="coefs", tol=1e-13, maxiter=500
        )
        s = float(r.scale)
        b = np.searchsorted(np.asarray(edges), x, side="right")
        loc = []
        for k in range(len(edges) + 1):
            yb = y[b == k]
            if len(yb) < 2:
                loc.append(float(yb[0]) if len(yb) else np.nan)
                continue
            rb = sm.RLM(yb, np.ones((len(yb), 1)), M=HuberT(DELTA)).fit(
                start_scale=s, update_scale=False, conv="coefs", tol=1e-13, maxiter=500
            )
            loc.append(float(rb.params[0]))
    return np.asarray(loc), s


def gaps(df: pl.DataFrame, cuts: list[int], **kw) -> dict[int, float]:
    """The worst bin's distance from the oracle over the rows so far, in the
    target's scales, after each row count."""
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    out = {}
    for n, (loc, edges) in locations(df, cuts, **kw).items():
        want, s = oracle(y[:n], x[:n], edges)
        out[n] = float(np.nanmax(np.abs(loc - want))) / s
    return out


def test_a_captured_stream_is_freed_on_the_refits_row():
    """Seed 138 at 20% spikes (the first rows of its 20,000-row stream, as
    the grid below draws it) is the worst of 50 without the refit: a bin 10
    scales from statsmodels at 500 rows and 3.5 at 2,000 (measured
    2026-10-26). The refit runs on the 500th row: there every bin is within
    the sketch's resolution of the oracle (measured 0.0012), and the one
    pass from there stays near it (0.045 at 1,000, 0.027 at 2,000), where
    `bin_refit_rows=0` -- the one pass alone, as 1.3.0 -- stays captured."""
    df = stream(20_000, 0.2, 138).head(2_000)
    cuts = [499, 500, 1_000, 2_000]
    on, off = gaps(df, cuts), gaps(df, cuts, bin_refit_rows=0)
    assert off[499] > 5.0 and off[2_000] > 2.0, f"the capture: {off}"
    assert on[499] == pytest.approx(off[499], rel=1e-12), "nothing moves before the refit"
    assert on[500] < 0.01, on
    assert on[1_000] < 0.1 and on[2_000] < 0.06, on


def test_learned_edges_refit_when_the_edges_are_fixed():
    """With learned edges (16 bins from 1,000 rows by default) the default
    count, 500, is within the warm-up, so the refit runs on the 1,000th row
    over every held row, and the bins never show a one-pass location: on
    that row every bin is within the sketch's resolution of statsmodels."""
    df = stream(20_000, 0.2, 138).head(1_000)
    on = gaps(df, [1_000], bin_edges=None, bins=16)
    off = gaps(df, [1_000], bin_edges=None, bins=16, bin_refit_rows=0)
    assert on[1_000] < 0.01 and off[1_000] > 1.0, (on, off)


def test_chunk_invariance_and_a_save_before_at_and_after_the_refit(tmp_path):
    """Hard rule 3 through the refit, with a decay, weights of 0 among the
    rows (which neither count nor teach) and given edges; and a bank saved
    while the rows are held, on the refit's row and after it, then loaded,
    lands where the uninterrupted one does, to the bit."""
    df = stream(1_500, 0.12, 9).with_columns(
        w=pl.Series(np.where(np.arange(1_500) % 7 == 3, 0.0, 1.0))
    )
    kw = dict(half_life=400.0, weight="w", bin_refit_rows=300)
    whole = po.ModelBank([spec(**kw)])
    whole.fit_predict(df)
    want = whole.marginal("m")
    for chunks in [7, 600]:
        bank = po.ModelBank([spec(**kw)])
        for part in df.iter_slices(-(-len(df) // chunks)):
            bank.fit_predict(part)
        assert bank.marginal("m").equals(want), f"{chunks} chunks"
    # The 300th learned row is row 349 (one row in seven has no weight).
    for at in [200, 350, 900]:
        bank = po.ModelBank([spec(**kw)])
        bank.fit_predict(df[:at])
        bank.save(tmp_path / f"m{at}.state")
        resumed = po.ModelBank.load(tmp_path / f"m{at}.state")
        resumed.fit_predict(df[at:])
        assert resumed.marginal("m").equals(want), f"saved at {at}"


def test_the_weights_do_not_move_and_zero_is_the_one_pass():
    """The refit moves locations and the clipped spreads read from them, and
    no weight: `bin_n`, `weight_sum` and every pair moment are the one
    pass's to the bit (CLAUDE.md rule 8). `bin_refit_rows=0` is the one pass
    itself, as 1.3.0 fitted it."""
    df = stream(1_200, 0.12, 4)
    on = po.ModelBank([spec()])
    off = po.ModelBank([spec(bin_refit_rows=0)])
    s_on, s_off = on.fit_predict(df), off.fit_predict(df)
    assert s_on.equals(s_off)
    t_on, t_off = on.marginal("m"), off.marginal("m")
    moved = ("bin_mean_y", "bin_var_y", "split_gain", "split_at", "split_gain_t")
    keep = [c for c in t_on.columns if c not in moved]
    assert t_on.select(keep).equals(t_off.select(keep))
    assert not t_on.select("bin_mean_y").equals(t_off.select("bin_mean_y"))


def test_the_default_count_and_the_resolved_spec():
    """`max(500, 30 * bins)` of the widest feature, and 0 -- off -- under the
    mean; a count given is taken."""

    def rows(**kw) -> int:
        resolved = json.loads(native.resolved_defaults(_spec._json(spec(**kw))))
        return resolved["model"]["bins"]["refit_rows"]

    assert rows() == 500
    assert rows(bin_edges=None, bins=32) == 960
    assert rows(bin_edges=[list(np.linspace(-1, 1, 41)[1:-1])]) == 1_200
    assert rows(bin_refit_rows=77) == 77
    assert rows(bin_stat="mean") == 0
    # Under `bin_huber_delta=inf` it is off too, which
    # `test_marginal_bin_huber.py`'s infinite cut holds to the mean's bits.


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        (dict(bin_stat="mean", bin_refit_rows=10), "bin_refit_rows is for the refit of"),
        (dict(bin_huber_delta=INF, bin_refit_rows=10), "bin_refit_rows is for the refit of"),
        (dict(bin_stat="mean", refit_budget=10.0), "refit_budget is for the refit of"),
        (dict(refit_budget=0.0), "refit_budget must be > 0 MiB"),
        (dict(bin_refit_rows=-1), "bin_refit_rows"),
    ],
)
def test_refusals(kw, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        po.ModelBank([spec(**kw)]).fit_predict(stream(50, 0.0, 0))


def test_the_refit_budget_bounds_the_groups_holds():
    """Given edges hold `bin_refit_rows` rows a group until the refit: under
    a `refit_budget` that admits three groups' holds, the fourth to appear
    and every later one keep the one pass, one notice says so naming
    `bin_refit_rows`, and `summary()` counts their targets. An admitted
    group is the unbounded bank's to the bit, a given-up one the bank's
    without the refit."""
    groups, rows = 6, 400
    parts = [stream(rows, 0.12, 20 + g).with_columns(g=pl.lit(str(g))) for g in range(groups)]
    df = pl.concat([p.with_columns(t=pl.col("t") * groups + i) for i, p in enumerate(parts)]).sort(
        "t"
    )
    # A hold of 300 rows of one feature and one target, 88 bytes each: the
    # row (two vectors, its decay and weight, 64), its feature (8) and its
    # target (an optional double, 16).
    budget = 3.5 * 300 * 88 / 2**20

    def bank(**kw) -> tuple[po.ModelBank, list[str]]:
        b = po.ModelBank([spec(group="g", bin_refit_rows=300, **kw)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            b.fit_predict(df)
        return b, [str(w.message) for w in caught if "refit_budget" in str(w.message)]

    bounded, notices = bank(refit_budget=budget)
    free, none = bank(refit_budget=INF)
    off = po.ModelBank([spec(group="g", bin_refit_rows=0)])
    off.fit_predict(df)
    assert len(notices) == 1 and "bin_refit_rows" in notices[0], notices
    assert none == []
    given = dict(bounded.summary().select("group", "refit_given_up").iter_rows())
    assert given == {str(g): (0 if g < 3 else 1) for g in range(groups)}, given
    for g in range(groups):
        mine = bounded.marginal("m", group=str(g))
        like = (free if g < 3 else off).marginal("m", group=str(g))
        assert mine.equals(like), g


#: The worst bin's distance from statsmodels, in target scales, over 50
#: seeds, after the rows in the key, at each share of spikes: the bound and,
#: in the comment, what was measured with the refit (and without it) on
#: 2026-10-26. 16 bins of given edges, the default count of 500.
_GRID_BOUNDS = {
    # 5%: 0.009 / 0.005 / 0.001 (without: 1.09 / 0.36 / 0.03)
    0.05: {1_000: 0.02, 2_000: 0.02, 20_000: 0.005},
    # 12%: 0.017 / 0.015 / 0.004 (without: 1.85 / 1.29 / 0.23)
    0.12: {1_000: 0.04, 2_000: 0.04, 20_000: 0.01},
    # 20%: 0.151 / 0.139 / 0.057 (without: 6.5 / 3.5 / 1.0)
    0.2: {1_000: 0.3, 2_000: 0.3, 20_000: 0.12},
}


@pytest.mark.oracle_grid(reason="a grid of 50 seeds against statsmodels' RLM")
@pytest.mark.parametrize("frac", sorted(_GRID_BOUNDS))
def test_no_seed_of_fifty_stays_captured(frac):
    """By the worst of 50 seeds, never a median: every bin within the bound
    of statsmodels' M-location at 1,000, 2,000 and 20,000 rows, and the
    worst falling from 2,000 rows to 20,000 -- the rate, not only the
    endpoint. The same seeds without the refit are held captured, so the
    grid shows the path it guards was taken."""
    cuts = sorted(_GRID_BOUNDS[frac])
    worst = dict.fromkeys(cuts, 0.0)
    worst_off = 0.0
    for seed in range(100, 150):
        df = stream(cuts[-1], frac, seed)
        for n, g in gaps(df, cuts).items():
            worst[n] = max(worst[n], g)
        worst_off = max(worst_off, gaps(df, [1_000], bin_refit_rows=0)[1_000])
    for n in cuts:
        assert worst[n] <= _GRID_BOUNDS[frac][n], f"{frac:.0%} at {n}: {worst}"
    assert worst[20_000] < worst[2_000], worst
    assert worst_off > 1.0, f"without the refit no seed was captured: {worst_off}"
