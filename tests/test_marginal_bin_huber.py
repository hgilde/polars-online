"""`marginal(bin_stat="huber")`: each bin's location as Huber's M-estimate of
location, learned in one pass, with the cut at `bin_huber_delta` times the
target's robust scale.

A screen that reads binned means of a residual, or of its square, sees a
shape wherever one spike lands in a thin bin: the mean moves by the spike's
size times its share. The robust location moves by at most the cut times
its share. The second opinion is statsmodels' batch M-location of each bin,
`statsmodels.robust.norms.estimate_location` with `HuberT`, at the
target's MAD (docs/PLAN.md task 272 (c)).
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest
from statsmodels.robust import norms
from statsmodels.robust.scale import mad

import polars_online as po

TIER = "essential"

DELTA = 1.345
#: Sixteen bins of equal width over the feature's range.
EDGES = [float(e) for e in np.linspace(-1.0, 1.0, 17)[1:-1]]


def stream(n: int, frac: float, seed: int, spike: float = 50.0) -> pl.DataFrame:
    """A curve the bins should trace, noise of 0.5, and one-sided spikes of
    `spike` on a share `frac` of the rows; `y_clean` is the row without its
    spike."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    clean = 2.0 * np.sin(3.0 * x) + 0.5 * rng.standard_normal(n)
    y = clean + np.where(rng.random(n) < frac, spike, 0.0)
    return pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y, "y_clean": clean})


def spec(**kw):
    d = dict(
        targets=["y"],
        features=["x"],
        clock="t",
        half_life=float("inf"),
        gap_cap=1e12,
        min_weight=2.0,
        bin_edges=[EDGES],
        bin_stat="huber",
    )
    d.update(kw)
    return po.spec.marginal("m", **d)


def table(df: pl.DataFrame, chunks: int = 1, **kw) -> pl.DataFrame:
    bank = po.ModelBank([spec(**kw)])
    parts = [df] if chunks == 1 else list(df.iter_slices(-(-len(df) // chunks)))
    for part in parts:
        bank.fit_predict(part)
    return bank.marginal("m")


def bins_of(df: pl.DataFrame) -> np.ndarray:
    return np.searchsorted(np.asarray(EDGES), df["x"].to_numpy(), side="right")


def batch_locations(df: pl.DataFrame) -> tuple[np.ndarray, float]:
    """statsmodels' converged M-location of each bin, at the target's MAD
    about its own M-location: the fixed point the one pass aims at."""
    y = df["y"].to_numpy()
    centre = float(norms.estimate_location(y, mad(y), norms.HuberT(DELTA), maxiter=500, tol=1e-13))
    s = float(mad(y, center=centre))
    b = bins_of(df)
    loc = [
        float(norms.estimate_location(y[b == k], s, norms.HuberT(DELTA), maxiter=500, tol=1e-13))
        for k in range(len(EDGES) + 1)
    ]
    return np.asarray(loc), s


@pytest.mark.parametrize(
    ("frac", "tol"), [(0.0, 0.002), (0.01, 0.003), (0.05, 0.004), (0.12, 0.06)]
)
def test_one_pass_lands_on_statsmodels_batch_m_location(frac, tol):
    """No decay, 20,000 rows in 16 bins (about 1,250 a bin): each bin's
    location is the converged batch M-location of its rows to within `tol`
    of the target's scale, over five seeds. The gap is the one pass's: early
    rows weighed against a location and a scale still settling. The mean,
    for contrast, sits `frac` times the spike away."""
    worst, worst_mean = 0.0, 0.0
    for seed in range(5):
        df = stream(20_000, frac, seed)
        got = np.asarray(table(df).row(0, named=True)["bin_mean_y"])
        want, s = batch_locations(df)
        worst = max(worst, float(np.max(np.abs(got - want))) / s)
        mean = np.asarray(table(df, bin_stat="mean").row(0, named=True)["bin_mean_y"])
        worst_mean = max(worst_mean, float(np.max(np.abs(mean - want))) / s)
    assert worst <= tol, f"{frac:.0%}: {worst:.5f} of the scale"
    if frac > 0:
        assert worst_mean > 20 * worst, f"the mean: {worst_mean:.4f}, huber {worst:.5f}"


def test_the_breakdown_at_one_five_and_twelve_percent():
    """Against the bins' clean means -- the curve the spikes hide -- the
    mean is off by about the share of spikes times their size, and the
    robust location by Huber's own bias, the cut over the curve's slope at
    the share's rows: a twentieth of the mean's at 1%, under a tenth at 12%."""
    for frac, most in [(0.01, 0.06), (0.05, 0.2), (0.12, 0.5)]:
        df = stream(20_000, frac, 7)
        b = bins_of(df)
        clean = np.array([df["y_clean"].to_numpy()[b == k].mean() for k in range(len(EDGES) + 1)])
        huber = np.asarray(table(df).row(0, named=True)["bin_mean_y"])
        mean = np.asarray(table(df, bin_stat="mean").row(0, named=True)["bin_mean_y"])
        err_h, err_m = np.max(np.abs(huber - clean)), np.max(np.abs(mean - clean))
        assert err_h < most, f"{frac:.0%}: huber {err_h:.3f}"
        assert err_m > 10 * err_h, f"{frac:.0%}: mean {err_m:.3f}, huber {err_h:.3f}"


def test_a_spike_in_a_thin_bin():
    """Twenty rows near 0 in one bin among 400 rows split between -5 and +5,
    then one row of 1,000 in it: the mean moves by about 1,000 / 21, the
    location by at most the cut over twenty (`c * w / W`, every earlier row
    counted in full), and the count by the row."""
    rng = np.random.default_rng(3)
    n = 420
    x = np.where(np.arange(n) % 21 == 0, 0.0, np.where(np.arange(n) % 2 == 0, -0.9, 0.9))
    y = np.where(x == 0.0, 0.0, np.sign(x) * 5.0) + 0.01 * rng.standard_normal(n)
    df = pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y})
    spike = pl.DataFrame({"t": [float(n)], "x": [0.0], "y": [1000.0]})
    kw = dict(bin_edges=[[-0.5, 0.5]])
    out = {}
    for stat in ["mean", "huber"]:
        bank = po.ModelBank([spec(bin_stat=stat, **kw)])
        bank.fit_predict(df)
        before = bank.marginal("m").row(0, named=True)
        bank.fit_predict(spike)
        after = bank.marginal("m").row(0, named=True)
        out[stat] = (before["bin_mean_y"][1], after["bin_mean_y"][1], after["bin_n"][1])
    m0, m1, n_mean = out["mean"]
    h0, h1, n_huber = out["huber"]
    assert m1 - m0 == pytest.approx((1000.0 - m0) / 21.0, rel=1e-12)
    assert n_mean == n_huber == 21.0
    # The target's scale: the median distance from its location, near 5,
    # over 0.6745; the cut is 1.345 of it, about 10.
    cut_most = 1.345 * 5.1 / 0.6744897501960817
    assert 0.0 < h1 - h0 <= cut_most / 20.0, (h0, h1)
    assert h1 - h0 > 0.4, "the row was clipped at the cut, not dropped"


def test_the_default_is_the_mean_and_an_infinite_cut_is_too():
    """`bin_stat` left out is the mean, the numbers of the build before it to
    the bit; `"huber"` with `bin_huber_delta=inf` clips nothing and is the
    mean to the bit too. Every field of the table."""
    df = stream(3_000, 0.05, 1)
    kw = dict(half_life=200.0, bin_edges=None, bins=8, bin_warm_rows=300)
    plain = table(df, **{**kw, "bin_stat": "mean"})
    left_out = po.ModelBank(
        [
            po.spec.marginal(
                "m",
                targets=["y"],
                features=["x"],
                clock="t",
                half_life=200.0,
                gap_cap=1e12,
                min_weight=2.0,
                bins=8,
                bin_warm_rows=300,
            )
        ]
    )
    left_out.fit_predict(df)
    assert left_out.marginal("m").equals(plain)
    assert table(df, **kw, bin_huber_delta=float("inf")).equals(plain)
    clipped = table(df, **kw)
    assert not clipped.equals(plain), "a finite cut clips the spikes"


def test_the_count_and_the_weight_are_the_means():
    """The rows' weight is what it is whatever the location (CLAUDE.md rule
    8): `bin_n`, `weight_sum` in the struct and every pair moment are the
    mean's to the bit, with weights, zero weights among them, and decay."""
    df = stream(2_000, 0.05, 2).with_columns(
        w=pl.Series(np.where(np.arange(2_000) % 7 == 3, 0.0, 1.0 + (np.arange(2_000) % 3)))
    )
    outs = {}
    for stat in ["mean", "huber"]:
        bank = po.ModelBank([spec(bin_stat=stat, weight="w", half_life=150.0)])
        outs[stat] = (bank.fit_predict(df), bank.marginal("m"))
    (s_mean, t_mean), (s_huber, t_huber) = outs["mean"], outs["huber"]
    assert s_mean.equals(s_huber), "the struct's weight_sum"
    keep = [
        c
        for c in t_mean.columns
        if c not in ("bin_mean_y", "bin_var_y", "split_gain", "split_at", "split_gain_t")
    ]
    assert t_mean.select(keep).equals(t_huber.select(keep))


def test_chunk_invariance():
    df = stream(4_000, 0.05, 4)
    kw = dict(half_life=300.0, bin_edges=None, bins=8, bin_warm_rows=200)
    whole = table(df, **kw)
    for chunks in [7, 600]:
        assert table(df, chunks=chunks, **kw).equals(whole), f"{chunks} chunks"


def test_a_saved_bank_resumes_to_the_bit(tmp_path):
    """Saved after the warm-up, with each target's location and scale and
    every cell's robust weight in the state, and loaded: the rest of the
    stream lands where the uninterrupted bank does."""
    df = stream(3_000, 0.05, 5)
    kw = dict(half_life=300.0, bin_edges=None, bins=8, bin_warm_rows=200)
    whole = table(df, **kw)
    bank = po.ModelBank([spec(**kw)])
    bank.fit_predict(df[:1_100])
    bank.save(tmp_path / "m.state")
    resumed = po.ModelBank.load(tmp_path / "m.state")
    resumed.fit_predict(df[1_100:])
    assert resumed.marginal("m").equals(whole)


def test_rows_of_weight_zero_age_and_teach_nothing():
    """`embargo`'s doubled stream: each row once at weight 0 when it
    arrives and again at its weight when its label matures. A row of weight
    0 ages the bins and the targets' scales and moves no location (CLAUDE.md
    rule 9), so the two banks agree to the bit."""
    df = stream(1_500, 0.05, 6)
    kw = dict(half_life=200.0, bin_edges=None, bins=8, bin_warm_rows=300)
    native = po.ModelBank([spec(embargo=7.0, **kw)])
    native.fit_predict(df)
    doubled = po.ModelBank([spec(weight=po.stream.ROLE + "_weight", **kw)])
    twice = po.stream.embargo(df, clock="t", delay=7.0)
    doubled.fit_predict(twice.filter(pl.col("t") <= df["t"].max()))
    assert native.marginal("m").equals(doubled.marginal("m"))


def test_the_spec_reports_back_what_it_was_given():
    """The bank's spec is the dict the builder made, and a spec left at the
    mean writes neither key, so its bytes are what they were."""
    s = spec(bin_huber_delta=2.0)
    assert po.ModelBank([s]).specs[0] == s
    assert s["model"]["bin_stat"] == "huber" and s["model"]["bin_huber_delta"] == 2.0
    plain = spec(bin_stat="mean")
    assert po.ModelBank([plain]).specs[0] == plain
    assert "bin_stat" not in plain["model"] and "bin_huber_delta" not in plain["model"]


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        (dict(bin_edges=None), "bin_stat is each bin's location and needs bins or bin_edges"),
        (dict(bin_stat="mean", bin_huber_delta=2.0), "bin_huber_delta is the cut of"),
        (dict(bin_huber_delta=0.0), "bin_huber_delta must be > 0"),
        (dict(bin_huber_delta=-1.0), "bin_huber_delta must be > 0"),
        (dict(bin_huber_delta=float("nan")), "bin_huber_delta must not be NaN"),
        (dict(bin_stat="median"), "bin_stat must be one of"),
    ],
)
def test_refusals(kw, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        table(stream(50, 0.0, 0), **kw)
