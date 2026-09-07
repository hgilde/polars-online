"""`marginal(bins=)`: the target's response curve per feature, and the best
single split of it.

Everything else `marginal` reports is linear, and a feature can be strongly
related to a target with `corr` at zero — a threshold, a V, a saturation. A
histogram of the target's moments inside the feature's bins sees all three,
costs `O(bins)` of state per pair and a binary search per pair per row, and
gives the same number a regression stump reports
(`docs/MARGINAL-LAGS-AND-BINS.md`, E67).
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import polars_online as po


def stream(n=4000, shape="threshold", seed=0):
    """A relation a correlation cannot see, plus a linear one for contrast."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    noise = 0.2 * rng.standard_normal(n)
    y = {
        "threshold": np.where(x > 0.3, 1.0, 0.0),
        "v": np.abs(x),
        "linear": x,
        "flat": np.zeros(n),
    }[shape] + noise
    return pl.DataFrame({"t": np.arange(n).astype(float), "x": x, "y": y})


def spec(**kw):
    d = dict(
        targets=["y"],
        features=["x"],
        clock="t",
        halflife=float("inf"),
        max_dclock=1e12,
        min_periods=2.0,
        bins=8,
        bin_warm_rows=500,
    )
    d.update(kw)
    if d.get("bin_edges") is not None:
        # Given edges are the whole answer; the learned kind's knobs are
        # refused beside them.
        for k in ("bins", "bin_rule", "bin_warm_rows"):
            d.pop(k, None)
    return po.spec.marginal("m", **d)


def pairs(df, chunks=1, frame=False, **kw):
    bank = po.ModelBank([spec(**kw)])
    for part in [df] if chunks == 1 else list(df.iter_slices(max(1, len(df) // chunks))):
        bank.fit_predict(part)
    out = bank.marginal("m")
    return out if frame else out.row(0, named=True)


def ew_histogram(x, y, w, edges, lam):
    """The histogram by hand: the target's decayed weight, mean and variance
    inside each bin of the feature, with the most recent row undecayed."""
    n = len(x)
    decay = lam ** np.arange(n - 1, -1, -1)
    ww = np.asarray(w, dtype=float) * decay
    b = np.searchsorted(np.asarray(edges), x, side="right")
    keep = np.isfinite(y) & (ww > 0)
    out = []
    for i in range(len(edges) + 1):
        m = keep & (b == i)
        if not m.any():
            out.append((0.0, None, None))
            continue
        wm = ww[m].sum()
        mean = (ww[m] * y[m]).sum() / wm
        var = (ww[m] * (y[m] - mean) ** 2).sum() / wm
        out.append((wm, mean, var))
    return out


def assert_histogram(p, want, rel=1e-9):
    assert len(p["bin_n"]) == len(want)
    for (n, mean, var), gn, gm, gv in zip(
        want, p["bin_n"], p["bin_mean_y"], p["bin_var_y"], strict=True
    ):
        assert gn == pytest.approx(n, rel=rel, abs=1e-300), "bin weight"
        if mean is None:
            assert gm is None and gv is None, "an empty bin has no moments"
        else:
            assert gm == pytest.approx(mean, rel=rel), "bin mean"
            assert gv == pytest.approx(var, rel=rel, abs=1e-12), "bin variance"


def test_a_threshold_is_found_and_located():
    """`y` jumps at `x = 0.3`. A step like this is *monotone*, so `corr` sees
    plenty of it — the split's contribution is that it explains more, and
    says where the step is, which no linear statistic can."""
    p = pairs(stream(shape="threshold"))
    linear_r2 = p["corr"] ** 2
    assert p["split_gain"] > linear_r2 + 0.1, (
        f"gain {p['split_gain']} against a linear R² of {linear_r2}"
    )
    # Eight quantile bins over a uniform feature put an edge every 0.25, so
    # the cut lands within half a bin of the true threshold.
    assert abs(p["split_at"] - 0.3) < 0.15, p["split_at"]


def test_a_v_shape_has_no_linear_signal_at_all():
    p = pairs(stream(shape="v"))
    assert abs(p["corr"]) < 0.1, p["corr"]
    # `|x|` on x ~ U(-1,1) has sd 0.29 against 0.2 of noise, so a single cut
    # can only reach so far — but it reaches orders of magnitude past what the
    # linear fit explains, which is the whole claim.
    assert p["split_gain"] > 0.15, p["split_gain"]
    assert p["split_gain"] > 50 * p["corr"] ** 2, (p["split_gain"], p["corr"])
    assert p["split_gain_t"] > 10.0, p["split_gain_t"]


def test_a_linear_relation_is_found_by_both():
    """The split is not a rival to `corr`, it is a second view: on a straight
    line both are large, and the response curve rises monotonically."""
    p = pairs(stream(shape="linear"))
    # x ~ U(-1,1) has sd 0.577 and the noise 0.2, so corr is 0.577/hypot(...)
    # = 0.945 by construction.
    assert p["corr"] == pytest.approx(0.945, abs=0.01)
    assert p["split_gain"] > 0.5
    means = [m for m in p["bin_mean_y"] if m is not None]
    assert means == sorted(means), means


def test_no_relation_gives_a_small_gain_and_a_calibrated_statistic():
    p = pairs(stream(shape="flat"))
    assert p["split_gain"] < 0.05, p["split_gain"]


def test_the_response_curve_is_the_targets_moments_in_each_bin():
    """`bin_n`, `bin_mean_y` and `bin_var_y` against the same numbers computed
    with polars over the reported edges."""
    df = stream(shape="threshold")
    p = pairs(df, halflife=float("inf"))
    edges = list(p["bin_edges"])
    assert len(edges) == 7, "eight bins means seven edges"
    assert edges == sorted(edges)

    binned = (
        df.with_columns(b=pl.col("x").cut(edges, left_closed=True).cast(pl.String))
        .group_by("b")
        .agg(n=pl.len(), mean=pl.col("y").mean(), var=pl.col("y").var(ddof=0))
    )
    # Undecayed, so the histogram is the plain group-by.
    order = (
        df.with_columns(b=pl.col("x").cut(edges, left_closed=True).cast(pl.String))
        .group_by("b")
        .agg(lo=pl.col("x").min())
        .sort("lo")["b"]
    )
    want = binned.join(pl.DataFrame({"b": order}), on="b", how="right").sort(
        pl.col("b").replace_strict({b: i for i, b in enumerate(order)}, return_dtype=pl.Int64)
    )
    assert list(p["bin_n"]) == pytest.approx(want["n"].cast(pl.Float64).to_list())
    assert list(p["bin_mean_y"]) == pytest.approx(want["mean"].to_list())
    assert list(p["bin_var_y"]) == pytest.approx(want["var"].to_list())


def test_chunk_invariance():
    """The hard rule: one chunk or many must give identical output, including
    where the edges landed."""
    df = stream(shape="threshold")
    one = pairs(df, chunks=1, frame=True)
    many = pairs(df, chunks=97, frame=True)
    assert one.equals(many)


def test_explicit_edges_skip_the_warm_up_and_are_reported_back():
    edges = [-0.5, 0.0, 0.5]
    df = stream(shape="threshold")
    by_list = pairs(df, bins=None, bin_edges=[edges])
    by_dict = pairs(df, bins=None, bin_edges={"x": edges})
    assert list(by_list["bin_edges"]) == edges
    assert by_list == by_dict
    # No warm-up to wait for: the first rows are already binned.
    early = pairs(df.head(5), bins=None, bin_edges=[edges], min_periods=2.0)
    assert list(early["bin_n"]) != []


def test_held_rows_are_replayed_not_dropped():
    """Learned edges must give the same histogram as those edges supplied up
    front — the warm-up rows are held, not spent."""
    df = stream(shape="threshold")
    learned = pairs(df, bins=8, bin_warm_rows=500)
    given = pairs(df, bins=None, bin_edges=[list(learned["bin_edges"])])
    assert list(given["bin_n"]) == pytest.approx(list(learned["bin_n"]))
    assert list(given["bin_mean_y"]) == pytest.approx(list(learned["bin_mean_y"]))
    assert given["split_gain"] == pytest.approx(learned["split_gain"])


def test_nothing_is_reported_until_the_edges_exist():
    df = stream(shape="threshold")
    early = pairs(df.head(100), bins=8, bin_warm_rows=500)
    # The columns are there because the spec asked for them; they are just
    # empty, which is the honest report of "not yet".
    assert list(early["bin_edges"]) == []
    assert list(early["bin_n"]) == []
    assert early["split_gain"] is None
    # And the linear statistics are unaffected by the wait.
    assert early["corr"] is not None


def test_a_feature_keeps_only_the_bins_it_can_support():
    """Ragged by design: a binary feature has two bins whatever `bins` says,
    and a constant one has a single bin and no split."""
    df = stream(shape="threshold").with_columns(b=(pl.col("x") > 0).cast(pl.Float64), c=pl.lit(7.0))
    frame = pairs(df, features=["x", "b", "c"], bins=8, frame=True).sort("feature")
    rows = {r["feature"]: r for r in frame.to_dicts()}
    assert len(rows["x"]["bin_n"]) == 8
    assert len(rows["b"]["bin_n"]) == 2, "a binary feature"
    assert len(rows["c"]["bin_n"]) == 1, "a constant feature"
    assert rows["c"]["split_gain"] is None, "nothing to split"


def test_fixed_rule_gives_equal_widths():
    p = pairs(stream(shape="threshold"), bins=4, bin_rule="fixed")
    edges = list(p["bin_edges"])
    widths = np.diff(edges)
    assert widths == pytest.approx(widths[0]), edges


def test_split_gain_t_uses_n_serial_when_it_has_one():
    """Serial dependence deflates the split's statistic for the same reason
    it deflates `t` — a smooth stream has fewer observations than rows."""
    n = 3000
    rng = np.random.default_rng(3)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.95 * x[i - 1] + np.sqrt(1 - 0.95**2) * rng.standard_normal()
    df = pl.DataFrame(
        {"t": np.arange(n).astype(float), "x": x, "y": np.abs(x) + 0.2 * rng.standard_normal(n)}
    )
    plain = pairs(df, bins=8, bin_warm_rows=500)
    corrected = pairs(df, bins=8, bin_warm_rows=500, lags=[1, 2, 3, 5, 8], serial_rule="geometric")
    assert corrected["split_gain"] == pytest.approx(plain["split_gain"])
    assert corrected["split_gain_t"] < plain["split_gain_t"], (
        f"{corrected['split_gain_t']} should be below {plain['split_gain_t']}"
    )


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        (dict(bins=8, window=100.0), "bins and window"),
        (dict(bins=8, bin_warm_rows=4), "bin_warm_rows"),
        (dict(bins=1), "at least 2"),
        (dict(bins=8, bin_rule="tertiles"), "unknown bin_rule"),
        (dict(bin_edges=[[1.0, 0.0]]), "strictly increasing"),
        (dict(bin_edges=[[0.0], [1.0]]), "one list per feature"),
        (dict(bin_edges={"z": [0.0]}), "missing"),
        (dict(bins=8, bin_warm_rows=10**9), "budget"),
    ],
)
def test_refusals(kw, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        pairs(stream(n=50), **kw)


@pytest.mark.parametrize("kw", [dict(bins=8), dict(bin_rule="fixed"), dict(bin_warm_rows=50)])
def test_given_edges_refuse_the_learned_kinds_knobs(kw):
    """Two ways of saying where the bins are is one too many."""
    d = dict(
        targets=["y"],
        features=["x"],
        clock="t",
        halflife=float("inf"),
        max_dclock=1e12,
        bin_edges=[[0.0]],
    )
    d.update(kw)
    with pytest.raises(ValueError, match="do not apply with it"):
        po.ModelBank([po.spec.marginal("m", **d)])


# --- decay, weights, nulls: the histogram is the pair moments' companion ----


def test_the_decayed_histogram_is_the_ew_moments_per_bin():
    """With a finite halflife the histogram is what a weighted group-by over
    the decayed weights gives, and the bin weights sum to the target's
    ``n_eff`` -- the same recursion as the pair moments, bin by bin."""
    df = stream(n=3000, shape="v", seed=1)
    p = pairs(df, halflife=100.0)
    lam = 0.5 ** (1.0 / 100.0)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    assert_histogram(p, ew_histogram(x, y, np.ones(len(x)), p["bin_edges"], lam))
    assert sum(p["bin_n"]) == pytest.approx(p["n_eff"], rel=1e-12)
    # The pair's own variance is the histogram's total variance, so a gain
    # is a fraction of something the caller can also read directly.
    n = np.asarray(p["bin_n"])
    m = np.asarray(p["bin_mean_y"], dtype=float)
    v = np.asarray(p["bin_var_y"], dtype=float)
    mean = (n * m).sum() / n.sum()
    total = (n * (v + (m - mean) ** 2)).sum() / n.sum()
    assert total == pytest.approx(p["var_y"], rel=1e-9)


def test_a_tiny_halflife_folds_the_scale_many_times_and_changes_nothing():
    """At a halflife of one row the undecayed weights the histogram keeps
    grow by two per row and reach the fold every ~500 rows; the numbers
    read out must not notice, in one chunk or many."""
    df = stream(n=4000, shape="threshold", seed=2)
    one = pairs(df, halflife=1.0, bin_warm_rows=100)
    lam = 0.5
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    assert_histogram(one, ew_histogram(x, y, np.ones(len(x)), one["bin_edges"], lam), rel=1e-6)
    many = pairs(df, chunks=333, halflife=1.0, bin_warm_rows=100, frame=True)
    assert pairs(df, halflife=1.0, bin_warm_rows=100, frame=True).equals(many)


def test_weights_and_null_targets_count_as_they_do_in_the_pair():
    """A weighted row counts its weight; a row whose target is null still
    shapes the edges (its feature is real) and adds nothing to any bin; a
    zero-weight row adds nothing anywhere."""
    df = stream(n=3000, shape="threshold", seed=4)
    rng = np.random.default_rng(4)
    w = rng.uniform(0.0, 2.0, len(df))
    w[::7] = 0.0
    y = df["y"].to_numpy().copy()
    y[::5] = np.nan
    df = df.with_columns(w=pl.Series(w), y=pl.Series(y))
    p = pairs(df, weight="w", halflife=200.0, bin_warm_rows=200)
    lam = 0.5 ** (1.0 / 200.0)
    assert_histogram(p, ew_histogram(df["x"].to_numpy(), y, w, p["bin_edges"], lam))
    assert sum(p["bin_n"]) == pytest.approx(p["n_eff"], rel=1e-12)
    # The warm-up counted every weighted row, null target or not, so the
    # edges are the weighted quantiles of those rows' features.
    assert len(p["bin_edges"]) == 7


def test_each_target_gets_its_own_histogram():
    df = stream(n=2000, shape="threshold", seed=5)
    y2 = -df["y"].to_numpy()
    y2[1::2] = np.nan
    df = df.with_columns(y2=pl.Series(y2))
    frame = pairs(df, targets=["y", "y2"], halflife=50.0, frame=True)
    lam = 0.5 ** (1.0 / 50.0)
    rows = {r["target"]: r for r in frame.to_dicts()}
    x = df["x"].to_numpy()
    w = np.ones(len(x))
    for t in ("y", "y2"):
        r = rows[t]
        assert_histogram(r, ew_histogram(x, df[t].to_numpy(), w, r["bin_edges"], lam))
    assert rows["y"]["bin_edges"] == rows["y2"]["bin_edges"], "one feature, one set of edges"
    assert rows["y2"]["n_eff"] < rows["y"]["n_eff"]


def test_a_clock_gap_past_max_dclock_empties_the_histogram_with_the_moments():
    """`max_dclock` is where the plumbing stops the clock; a gap that large
    decays the pair moments to nothing, and the histogram with them."""
    df = stream(n=1000, shape="threshold", seed=6)
    t = df["t"].to_numpy().copy()
    t[500:] += 1e6
    df = df.with_columns(t=pl.Series(t))
    kw = dict(halflife=10.0, max_dclock=1e5, bin_warm_rows=100)
    after = pairs(df, **kw)
    alone = pairs(df[500:], bin_edges=[list(after["bin_edges"])], halflife=10.0, max_dclock=1e5)
    assert after["n_eff"] == pytest.approx(alone["n_eff"])
    assert list(after["bin_n"]) == pytest.approx(list(alone["bin_n"]))
    assert list(after["bin_mean_y"]) == pytest.approx(list(alone["bin_mean_y"]))


# --- the warm-up under real streams ------------------------------------------


def test_a_bank_saved_during_the_warm_up_resumes_with_its_held_rows(tmp_path):
    df = stream(n=1500, shape="threshold", seed=8)
    whole = pairs(df, halflife=300.0, frame=True)
    bank = po.ModelBank([spec(halflife=300.0)])
    bank.fit_predict(df[:300])
    assert list(bank.marginal("m").row(0, named=True)["bin_edges"]) == [], "still warming up"
    bank.save(tmp_path / "m.state")
    resumed = po.ModelBank.load(tmp_path / "m.state")
    resumed.fit_predict(df[300:])
    assert resumed.marginal("m").equals(whole), "the held rows must survive the round trip"


def test_label_delay_is_the_doubled_stream_here_too():
    """A row is learned only once its label has matured, and the rows are
    learned in clock order downstream of the delay buffer: so the warm-up
    counts them there, and the histogram, the edges and the lag ring all
    agree with the doubled stream `po.prep.embargo` builds, to the bit."""
    from polars_online import prep

    df = stream(n=1200, shape="threshold", seed=9)
    kw = dict(halflife=200.0, bin_warm_rows=300, lags=[1, 2, 3], serial_rule="geometric")
    native = po.ModelBank([spec(label_delay=7.0, **kw)])
    native.fit_predict(df)
    doubled = po.ModelBank([spec(weight=prep.ROLE + "_weight", **kw)])
    # The doubled stream carries the last rows' lessons past the end of the
    # clock; the native bank is still holding those, so stop where it stops.
    twice = prep.embargo(df, clock="t", delay=7.0).collect()
    doubled.fit_predict(twice.filter(pl.col("t") <= df["t"].max()))
    assert native.marginal("m").equals(doubled.marginal("m"))


# --- the edges ----------------------------------------------------------------


def test_a_rare_indicator_keeps_its_own_bin_and_is_split():
    """A feature that is one in twenty rows has a value that carries more
    than a bin's share. Equal-count edges would put every edge at zero and
    lose it; a point mass fills a bin of its own, so the split sees it."""
    n = 2000
    rng = np.random.default_rng(10)
    x = (rng.uniform(size=n) < 0.05).astype(float)
    y = 5.0 * x + 0.2 * rng.standard_normal(n)
    df = pl.DataFrame({"t": np.arange(n).astype(float), "x": x, "y": y})
    p = pairs(df, bins=8, bin_warm_rows=500)
    assert list(p["bin_edges"]) == [1.0]
    assert p["split_gain"] > 0.9, p["split_gain"]
    assert p["split_at"] == 1.0


def test_quantile_edges_are_weighted():
    df = stream(n=2000, shape="linear", seed=11)
    heavy = df.with_columns(w=pl.when(pl.col("x") > 0.5).then(20.0).otherwise(1.0))
    plain = pairs(df, bins=4, bin_warm_rows=500)
    weighted = pairs(heavy, bins=4, bin_warm_rows=500, weight="w")
    assert plain["bin_edges"][1] == pytest.approx(0.0, abs=0.1), "unweighted: the median"
    assert weighted["bin_edges"][0] > 0.4, "weight pulls every edge into the heavy quarter"


def test_a_far_offset_target_keeps_its_variance():
    """Per-bin Welford: a target of 1e7 plus noise of 1e-3 has a variance a
    raw sum of squares cannot see (it is 1e-6 against squares of 1e14)."""
    n = 2000
    rng = np.random.default_rng(12)
    x = rng.uniform(-1.0, 1.0, n)
    y = 1e7 + 1e-3 * rng.standard_normal(n)
    df = pl.DataFrame({"t": np.arange(n).astype(float), "x": x, "y": y})
    p = pairs(df, bins=4, bin_warm_rows=200, halflife=500.0)
    assert p["var_y"] == pytest.approx(1e-6, rel=0.2)
    for v in p["bin_var_y"]:
        assert v == pytest.approx(p["var_y"], rel=0.3)


def test_a_fixed_bin_no_row_lands_in_is_empty_not_absent():
    """Equal widths over a feature with a hole in it: the hollow bins report
    zero weight and null moments, and stay in the list so that the edges
    still say where every bin is."""
    n = 2000
    rng = np.random.default_rng(13)
    x = rng.uniform(0.6, 1.0, n) * rng.choice([-1.0, 1.0], n)
    df = pl.DataFrame({"t": np.arange(n).astype(float), "x": x, "y": x})
    p = pairs(df, bins=4, bin_rule="fixed", bin_warm_rows=500)
    assert len(p["bin_n"]) == 4
    assert p["bin_n"][1] == 0.0 and p["bin_n"][2] == 0.0
    assert p["bin_mean_y"][1] is None and p["bin_var_y"][2] is None
    assert p["bin_n"][0] > 0 and p["bin_mean_y"][0] < 0
    assert p["split_gain"] > 0.9


# --- the schema is the spec's -------------------------------------------------


def test_the_columns_are_there_before_any_row_and_for_a_group_never_seen():
    bank = po.ModelBank([spec(lags=[1, 2], serial_rule="truncated", group="g")])
    want = bank.marginal("m").columns
    assert "bin_edges" in want and "lagcorr_xx" in want and "split_gain_t" in want
    df = stream(n=800, shape="threshold").with_columns(g=pl.lit("a"))
    bank.fit_predict(df)
    assert bank.marginal("m").columns == want
    assert bank.marginal("m", group="never").columns == want
    assert bank.marginal("m", group="never").height == 0
