"""E58: refresh-time sampling — asynchronous series on a common grid.

The oracle is a longhand Python loop over the same ticks: a grid point
wherever every series has ticked at least once since the last one. Everything
else here is the properties that make the sampler usable — nothing
interpolated, the same grid whatever the chunking, and a retained fraction
that says how much of the data survived.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import prep

NAMES = ["a", "b", "c"]


def poisson_ticks(names=NAMES, rates=(1.0, 0.4, 0.15), n=400, seed=0):
    """A long frame of ticks, each series at its own rate, in time order."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, rate in zip(names, rates, strict=True):
        t = np.cumsum(rng.exponential(1.0 / rate, n))
        rows.append(pl.DataFrame({"series": [s] * n, "t": t, "v": rng.standard_normal(n)}))
    return pl.concat(rows).sort("t")


def oracle(df, names, pairs=False):
    """The recursion longhand: last value per series, a point where the set
    completes, counts since the previous point."""
    if pairs:
        out = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                two = df.filter(pl.col("series").is_in([names[i], names[j]]))
                for row in oracle(two, [names[i], names[j]]):
                    out.append({"pair": f"{names[i]}|{names[j]}", **row})
        return sorted(out, key=lambda r: (r["time_refresh"], r["pair"]))
    last = dict.fromkeys(names, float("nan"))
    ticks = dict.fromkeys(names, 0)
    seen: set[str] = set()
    out = []
    for s, t, v in df.select("series", "t", "v").iter_rows():
        if v is None:
            continue
        last[s] = v
        ticks[s] += 1
        seen.add(s)
        if len(seen) == len(names):
            out.append(
                {
                    "time_refresh": t,
                    **{f"{n}_value": last[n] for n in names},
                    **{f"n_ticks_{n}": ticks[n] for n in names},
                    "retained_fraction": len(names) / sum(ticks.values()),
                }
            )
            seen = set()
            ticks = dict.fromkeys(names, 0)
    return out


def run(df, **kw):
    kw.setdefault("series", "series")
    kw.setdefault("names", NAMES)
    kw.setdefault("time", "t")
    kw.setdefault("value", "v")
    return prep.refresh_time(df, **kw).collect()


def test_the_grid_is_the_longhand_loop():
    df = poisson_ticks()
    got = run(df)
    want = pl.DataFrame(oracle(df, NAMES))
    assert got.equals(want)


def test_pairs_are_the_longhand_loop_per_pair():
    df = poisson_ticks()
    got = run(df, pairs=True).sort("time_refresh", "pair")
    want = oracle(df, NAMES, pairs=True)
    assert got.height == len(want)
    for row, w in zip(got.iter_rows(named=True), want, strict=True):
        assert row["pair"] == w["pair"]
        assert row["time_refresh"] == w["time_refresh"]
        a, b = w["pair"].split("|")
        assert row["a_value"] == w[f"{a}_value"]
        assert row["b_value"] == w[f"{b}_value"]
        assert row["n_ticks_a"] == w[f"n_ticks_{a}"]
        assert row["n_ticks_b"] == w[f"n_ticks_{b}"]


def test_pairs_keep_more_of_the_data_than_the_joint_grid():
    """The point of `pairs`: one slow series holds the joint grid to its own
    pace, and every pair not containing it keeps going."""
    df = poisson_ticks(rates=(2.0, 2.0, 0.05))
    joint = run(df)
    per_pair = run(df, pairs=True)
    ab = per_pair.filter(pl.col("pair") == "a|b").height
    assert ab > 5 * joint.height


def test_a_synchronous_input_is_returned_unchanged():
    n = 50
    rng = np.random.default_rng(1)
    df = pl.concat(
        [
            pl.DataFrame(
                {
                    "series": [s] * n,
                    "t": np.arange(float(n)),
                    "v": rng.standard_normal(n),
                }
            )
            for s in NAMES
        ]
    ).sort("t", "series")
    out = run(df)
    assert out.height == n
    for s in NAMES:
        assert out[f"n_ticks_{s}"].to_list() == [1] * n
    assert out["retained_fraction"].to_list() == [1.0] * n
    # And the values are the input's, not an interpolation of it.
    for s in NAMES:
        want = df.filter(pl.col("series") == s)["v"].to_list()
        assert out[f"{s}_value"].to_list() == want


def test_the_eight_nine_ten_example_gives_seven_points_and_21_of_27():
    """BNHLS's three-series picture. Their tick *times* are only a figure in
    the paper, so the stream is constructed to their counts (8, 9, 10) and
    the reduction is what is checked."""
    intervals = [
        ["c", "c", "c", "a", "b"],
        ["b", "b", "a", "c"],
        ["b", "b", "a", "c"],
        ["a", "a", "b", "c"],
        ["c", "c", "a", "b"],
        ["a", "b", "c"],
        ["a", "b", "c"],
    ]
    ticks = [s for iv in intervals for s in iv]
    assert [ticks.count(s) for s in NAMES] == [8, 9, 10]
    df = pl.DataFrame(
        {
            "series": ticks,
            "t": np.arange(1.0, len(ticks) + 1),
            "v": np.arange(1.0, len(ticks) + 1),
        }
    )
    out = run(df)
    assert out.height == 7, "N = 7"
    kept = 3 * out.height
    assert kept == 21 and len(ticks) == 27
    assert out["retained_fraction"].to_list() == [0.6, 0.75, 0.75, 0.75, 0.75, 1.0, 1.0]
    # N never exceeds the slowest series' tick count.
    assert out.height <= min(ticks.count(s) for s in NAMES)


@pytest.mark.parametrize("size", [1, 3, 97, 100_000])
def test_the_same_grid_from_one_chunk_and_from_a_thousand(size):
    df = poisson_ticks(n=200)
    want = run(df)
    got = prep.refresh_time(
        df, series="series", names=NAMES, time="t", value="v", chunk_rows=size
    ).collect()
    assert want.equals(got)


def test_groups_keep_their_own_grids():
    a = poisson_ticks(n=120, seed=2).with_columns(g=pl.lit("x"))
    b = poisson_ticks(n=120, seed=3).with_columns(g=pl.lit("y"))
    both = pl.concat([a, b]).sort("t")
    out = run(both, by="g")
    for key, part in (("x", a), ("y", b)):
        want = pl.DataFrame(oracle(part, NAMES))
        got = out.filter(pl.col("g") == key).drop("g")
        assert got.equals(want)


@pytest.mark.parametrize(
    "dtype", [pl.Int64, pl.UInt32, pl.String, pl.Categorical, pl.Enum(["x", "y"])]
)
def test_the_by_column_comes_back_in_the_dtype_it_went_in_as(dtype):
    """The keys are held as text inside, but the column goes back out in the
    dtype it came in as -- so the result joins to the frame it came from,
    and matches the schema the lazy plan declared
    (docs/REVIEW-E54-E64.md RT1)."""
    labels = [0, 1] if dtype in (pl.Int64, pl.UInt32) else ["x", "y"]
    parts = [
        poisson_ticks(n=60, seed=4 + i).with_columns(g=pl.lit(k).cast(dtype))
        for i, k in enumerate(labels)
    ]
    df = pl.concat(parts).sort("t")
    lazy = prep.refresh_time(df, series="series", names=NAMES, time="t", value="v", by="g")
    out = lazy.collect()
    assert out.schema["g"] == dtype
    assert lazy.collect_schema()["g"] == dtype
    assert sorted(out["g"].cast(pl.String).unique().to_list()) == sorted(str(k) for k in labels)


def test_a_tie_at_a_grid_point_belongs_to_the_next_interval():
    """ "Strictly after tau_j" is read against the row sequence: a tick with
    the same timestamp as the one that just closed a point, but later in the
    frame, starts the next interval. That is what lets a point be emitted
    where its last series ticks rather than held for a greater timestamp,
    and it is what makes the result chunk-invariant
    (docs/REVIEW-E54-E64.md RT4)."""
    df = pl.DataFrame(
        {
            "series": ["a", "b", "b", "a", "a", "b"],
            "t": [1.0, 1.0, 1.0, 2.0, 3.0, 3.0],
            "v": [1.0, 2.0, 2.5, 3.0, 4.0, 5.0],
        }
    )
    out = prep.refresh_time(df, series="series", names=["a", "b"], time="t", value="v").collect()
    # Three points. The first closes on b's tick at t = 1 with b = 2.0; b's
    # *second* tick at t = 1 is after that point, so it belongs to the next
    # interval -- which a then completes at t = 2, carrying b = 2.5, a
    # value observed at t = 1. Read the timestamp alone and there would be
    # two points, at t = 1 and t = 3.
    assert out["time_refresh"].to_list() == [1.0, 2.0, 3.0]
    assert out["b_value"].to_list() == [2.0, 2.5, 5.0]
    assert out["a_value"].to_list() == [1.0, 3.0, 4.0]


def test_keep_columns_take_the_completing_ticks_value():
    df = poisson_ticks(n=60).with_row_index("i").with_columns(pl.col("i").cast(pl.Int64))
    out = run(df, keep=["i"])
    # Every kept value is the row index of the tick that closed the point.
    times = df.select("t", "i")
    for t, i in out.select("time_refresh", "i").iter_rows():
        assert times.filter(pl.col("t") == t)["i"].to_list() == [i]


def test_a_null_value_is_a_tick_that_observed_nothing():
    df = pl.DataFrame(
        {
            "series": ["a", "b", "c", "b", "x"],
            "t": [1.0, 2.0, 3.0, 4.0, 5.0],
            "v": [1.0, None, 3.0, 4.0, 5.0],
        }
    ).head(4)
    out = run(df)
    assert out.height == 1
    assert out["time_refresh"][0] == 4.0 and out["b_value"][0] == 4.0
    assert out["n_ticks_b"][0] == 1, "the null tick is not counted"


def test_the_pushdowns_are_honoured():
    df = poisson_ticks(n=200)
    plan = prep.refresh_time(df, series="series", names=NAMES, time="t", value="v")
    full = plan.collect()
    assert plan.head(5).collect().equals(full.head(5))
    assert plan.select("time_refresh").collect().equals(full.select("time_refresh"))
    hot = plan.filter(pl.col("retained_fraction") > 0.5).collect()
    assert hot.equals(full.filter(pl.col("retained_fraction") > 0.5))


def test_the_output_feeds_a_bank_of_the_wide_frame():
    """What the grid is for: a correlation over columns that were never
    observed at the same time."""
    df = poisson_ticks(n=300)
    grid = run(df)
    spec = po.spec.ew_cov(
        "c",
        features=[f"{s}_value" for s in NAMES],
        halflife=100.0,
        stats=["corr"],
        min_periods=5.0,
    )
    out = po.ModelBank([spec]).fit_predict(grid)
    assert out["c"].struct.field("corr_a_value_b_value").drop_nulls().len() > 0


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"names": ["a"]}, "at least two series"),
        ({"names": ["a", "a"]}, "more than once"),
        ({"series": "nope"}, "no series column"),
        ({"time": "nope"}, "no time column"),
        ({"value": "nope"}, "no value column"),
        ({"by": "nope"}, "no by column"),
        ({"keep": ["nope"]}, "no keep column"),
    ],
)
def test_a_bad_call_is_refused_while_the_plan_is_built(kw, message):
    with pytest.raises(ValueError, match=message):
        run(poisson_ticks(n=10), **kw)


def test_an_unknown_series_and_a_backwards_time_are_refused_naming_the_row():
    df = pl.DataFrame({"series": ["a", "b", "z"], "t": [1.0, 2.0, 3.0], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(Exception, match="row 2.*'z'|row 2.*\"z\""):
        run(df, names=["a", "b"])
    back = pl.DataFrame({"series": ["a", "b", "a"], "t": [1.0, 5.0, 2.0], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(Exception, match="row 2.*time order"):
        run(back, names=["a", "b"])
