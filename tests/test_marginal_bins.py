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


def pairs(df, chunks=1, frame=False, **kw):
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
    bank = po.ModelBank([po.spec.marginal("m", **d)])
    for part in [df] if chunks == 1 else list(df.iter_slices(max(1, len(df) // chunks))):
        bank.fit_predict(part)
    out = bank.marginal("m")
    return out if frame else out.row(0, named=True)


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
        (dict(bins=None, bin_edges=[[1.0, 0.0]]), "strictly increasing"),
        (dict(bins=None, bin_edges=[[0.0], [1.0]]), "one list per feature"),
        (dict(bins=None, bin_edges={"z": [0.0]}), "missing"),
        (dict(bins=8, bin_warm_rows=10**9), "budget"),
    ],
)
def test_refusals(kw, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        pairs(stream(n=50), **kw)
