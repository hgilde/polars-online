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
from polars_online import stream

TIER = "mixed"

NAMES = ["a", "b", "c"]


def poisson_obs(names=NAMES, rates=(1.0, 0.4, 0.15), n=400, seed=0):
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
        # Missing by the rule every spec column follows: null, NaN, an
        # infinity, or a magnitude past the input bound of 1e100.
        if v is None or not abs(v) <= 1e100:
            continue
        last[s] = v
        ticks[s] += 1
        seen.add(s)
        if len(seen) == len(names):
            out.append(
                {
                    "time_refresh": t,
                    **{f"{n}_value": last[n] for n in names},
                    **{f"n_obs_{n}": ticks[n] for n in names},
                    "retained_fraction": len(names) / sum(ticks.values()),
                }
            )
            seen = set()
            ticks = dict.fromkeys(names, 0)
    return out


def run(df, **kw):
    kw.setdefault("series", "series")
    kw.setdefault("names", NAMES)
    kw.setdefault("clock", "t")
    kw.setdefault("value", "v")
    return stream.refresh_time(df, **kw)


def test_the_grid_is_the_longhand_loop():
    df = poisson_obs()
    got = run(df)
    want = pl.DataFrame(oracle(df, NAMES))
    assert got.equals(want)


def test_pairs_are_the_longhand_loop_per_pair():
    df = poisson_obs()
    got = run(df, pairs=True).sort("time_refresh", "pair")
    want = oracle(df, NAMES, pairs=True)
    assert got.height == len(want)
    for row, w in zip(got.iter_rows(named=True), want, strict=True):
        assert row["pair"] == w["pair"]
        assert row["time_refresh"] == w["time_refresh"]
        a, b = w["pair"].split("|")
        assert row["a_value"] == w[f"{a}_value"]
        assert row["b_value"] == w[f"{b}_value"]
        assert row["n_obs_a"] == w[f"n_obs_{a}"]
        assert row["n_obs_b"] == w[f"n_obs_{b}"]


def test_pairs_keep_more_of_the_data_than_the_joint_grid():
    """The point of `pairs`: one slow series holds the joint grid to its own
    pace, and every pair not containing it keeps going."""
    df = poisson_obs(rates=(2.0, 2.0, 0.05))
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
        assert out[f"n_obs_{s}"].to_list() == [1] * n
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


def test_the_readme_example_is_the_grid_it_shows():
    """The README's eight ticks and the table it prints of their grid, read
    off by hand: a point when the last series to tick ticks, each series'
    last value there, and its ticks since the previous point."""
    ticks = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "CCC", "BBB", "AAA", "AAA", "CCC"],
            "t": [0.4, 0.9, 1.3, 1.6, 2.2, 2.5, 2.8, 3.1],
            "px": [100.0, 20.0, 100.2, 50.0, 20.1, 100.1, 100.4, 49.9],
        }
    )
    grid = po.stream.refresh_time(
        ticks, series="symbol", names=["AAA", "BBB", "CCC"], clock="t", value="px"
    )
    assert grid.rows() == [
        (1.6, 100.2, 20.0, 50.0, 2, 1, 1, 0.75),
        (3.1, 100.4, 20.1, 49.9, 2, 1, 1, 0.75),
    ]
    assert grid.columns == [
        "time_refresh",
        "AAA_value",
        "BBB_value",
        "CCC_value",
        "n_obs_AAA",
        "n_obs_BBB",
        "n_obs_CCC",
        "retained_fraction",
    ]


@pytest.mark.parametrize("size", [1, 3, 97, 100_000])
def test_the_same_grid_from_one_chunk_and_from_a_thousand(size):
    df = poisson_obs(n=200)
    want = run(df)
    got = stream.refresh_time(
        df.lazy(), series="series", names=NAMES, clock="t", value="v", chunk_size=size
    ).collect()
    assert want.equals(got)


def test_groups_keep_their_own_grids():
    a = poisson_obs(n=120, seed=2).with_columns(g=pl.lit("x"))
    b = poisson_obs(n=120, seed=3).with_columns(g=pl.lit("y"))
    both = pl.concat([a, b]).sort("t")
    out = run(both, group="g")
    for key, part in (("x", a), ("y", b)):
        want = pl.DataFrame(oracle(part, NAMES))
        got = out.filter(pl.col("g") == key).drop("g")
        assert got.equals(want)


@pytest.mark.parametrize(
    "dtype", [pl.Int64, pl.UInt32, pl.String, pl.Categorical, pl.Enum(["x", "y"])]
)
def test_the_group_column_comes_back_in_the_dtype_it_went_in_as(dtype):
    """The keys are held as text inside, but the column goes back out in the
    dtype it came in as -- so the result joins to the frame it came from,
    and matches the schema the lazy plan declared
    (docs/REVIEW-E54-E64.md RT1)."""
    labels = [0, 1] if dtype in (pl.Int64, pl.UInt32) else ["x", "y"]
    parts = [
        poisson_obs(n=60, seed=4 + i).with_columns(g=pl.lit(k).cast(dtype))
        for i, k in enumerate(labels)
    ]
    df = pl.concat(parts).sort("t")
    lazy = stream.refresh_time(
        df.lazy(), series="series", names=NAMES, clock="t", value="v", group="g"
    )
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
    out = stream.refresh_time(df, series="series", names=["a", "b"], clock="t", value="v")
    # Three points. The first closes on b's tick at t = 1 with b = 2.0; b's
    # *second* tick at t = 1 is after that point, so it belongs to the next
    # interval -- which a then completes at t = 2, carrying b = 2.5, a
    # value observed at t = 1. Read the timestamp alone and there would be
    # two points, at t = 1 and t = 3.
    assert out["time_refresh"].to_list() == [1.0, 2.0, 3.0]
    assert out["b_value"].to_list() == [2.0, 2.5, 5.0]
    assert out["a_value"].to_list() == [1.0, 3.0, 4.0]


def test_keep_columns_take_the_completing_ticks_value():
    df = poisson_obs(n=60).with_row_index("i").with_columns(pl.col("i").cast(pl.Int64))
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
    assert out["n_obs_b"][0] == 1, "the null tick is not counted"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 1e101])
def test_a_value_the_bank_reads_as_missing_observed_nothing(bad):
    """NaN, an infinity and a magnitude past 1e100 are missing values, as
    they are in every spec column, so such a tick observed nothing (review
    round 4, PC7). They were values: a point completed on a NaN tick
    reported ``b_value`` NaN."""
    df = pl.DataFrame(
        {"series": ["a", "b", "c", "b"], "t": [1.0, 2.0, 3.0, 4.0], "v": [1.0, bad, 3.0, 4.0]}
    )
    out = run(df)
    assert out.height == 1
    assert out["time_refresh"][0] == 4.0 and out["b_value"][0] == 4.0
    assert out["n_obs_b"][0] == 1, "the missing tick is not counted"
    # And through the longhand loop, on a stream with them sprinkled in.
    ticks = poisson_obs(n=300)
    holed = ticks.with_columns(
        v=pl.when(pl.int_range(pl.len()) % 7 == 3).then(bad).otherwise(pl.col("v"))
    )
    assert run(holed).equals(pl.DataFrame(oracle(holed, NAMES)))


@pytest.mark.parametrize(
    "value",
    [
        pl.Series(["1.5", "2.5", "3.5"]),
        pl.Series(["abc", "def", "ghi"]),
        pl.Series([19000, 19001, 19002]).cast(pl.Date),
        pl.Series([1, 2, 3]).cast(pl.Datetime("us")),
        pl.Series([[1.0], [2.0], [3.0]]),
    ],
    ids=["numbers-as-text", "text", "date", "datetime", "list"],
)
def test_a_value_column_that_is_not_numeric_is_refused_by_name(value):
    """Refused while the plan is built, as a spec refuses a feature of that
    dtype (review round 4, PC7). It was cast without a check: text came out
    all null and the grid was empty with nothing said, and a ``Date`` was
    read as its day count."""
    df = pl.DataFrame({"series": ["a", "b", "c"], "t": [1.0, 2.0, 3.0], "v": value})
    with pytest.raises(ValueError, match="value column 'v' has dtype .*; it must be numeric"):
        run(df)


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Float32, pl.Boolean, pl.UInt8, pl.Null])
def test_a_value_column_of_any_numeric_dtype_is_read(dtype):
    """Integers, booleans and a column of nulls are numbers, as they are to a
    spec."""
    v = (
        pl.Series([None] * 3, dtype=pl.Null)
        if dtype == pl.Null
        else pl.Series([1, 0, 1]).cast(dtype)
    )
    df = pl.DataFrame({"series": ["a", "b", "c"], "t": [1.0, 2.0, 3.0], "v": v})
    out = run(df)
    assert out.height == (0 if dtype == pl.Null else 1)


@pytest.mark.parametrize(
    "clock",
    [
        pl.Series(["1.5", "2.5", "3.5"]),
        pl.Series(["abc", "def", "ghi"]),
        pl.Series(["1", "2", "3"]).cast(pl.Categorical),
        pl.Series([[1.0], [2.0], [3.0]]),
    ],
    ids=["numbers-as-text", "text", "categorical", "list"],
)
def test_a_clock_column_that_is_neither_numeric_nor_temporal_is_refused_by_name(clock):
    """Refused while the plan is built, as the bank refuses a clock of that
    dtype (task 187's rule for ``value``, PC7, applied to the clock). It was
    cast without a check: text of digits was read as a clock, and other
    text was refused as a null clock on its first row."""
    df = pl.DataFrame({"series": ["a", "b", "c"], "t": clock, "v": [1.0, 2.0, 3.0]})
    with pytest.raises(
        ValueError, match="clock column 't' has dtype .*; it must be numeric or temporal"
    ):
        run(df)


@pytest.mark.parametrize(
    "dtype", [pl.Int64, pl.Float32, pl.UInt8, pl.Boolean, pl.Datetime("ms"), pl.Duration("us")]
)
def test_a_clock_column_the_bank_reads_is_read(dtype):
    """A number, a boolean and a temporal column are clocks, as they are to a
    spec."""
    t = pl.Series([0, 1, 1]).cast(dtype)
    df = pl.DataFrame({"series": ["a", "b", "c"], "t": t, "v": [1.0, 2.0, 3.0]})
    out = run(df)
    assert out.height == 1 and out.schema["time_refresh"] == dtype


def test_the_pushdowns_are_honoured():
    df = poisson_obs(n=200)
    plan = stream.refresh_time(df.lazy(), series="series", names=NAMES, clock="t", value="v")
    full = plan.collect()
    assert plan.head(5).collect().equals(full.head(5))
    assert plan.select("time_refresh").collect().equals(full.select("time_refresh"))
    hot = plan.filter(pl.col("retained_fraction") > 0.5).collect()
    assert hot.equals(full.filter(pl.col("retained_fraction") > 0.5))


def test_the_output_feeds_a_bank_of_the_wide_frame():
    """What the grid is for: a correlation over columns that were never
    observed at the same time."""
    df = poisson_obs(n=300)
    grid = run(df)
    spec = po.spec.ew_cov(
        "c",
        features=[f"{s}_value" for s in NAMES],
        half_life=100.0,
        stats=["corr"],
        min_weight=5.0,
    )
    out = po.ModelBank([spec]).fit_predict(grid)
    assert out["c"].struct.field("corr_a_value_b_value").drop_nulls().len() > 0


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"names": ["a"]}, "at least two series"),
        ({"names": ["a", "a"]}, "more than once"),
        ({"series": "nope"}, "no series column"),
        ({"clock": "nope"}, "no clock column"),
        ({"value": "nope"}, "no value column"),
        ({"group": "nope"}, "no group column"),
        ({"keep": ["nope"]}, "no keep column"),
    ],
)
def test_a_bad_call_is_refused_while_the_plan_is_built(kw, message):
    with pytest.raises(ValueError, match=message):
        run(poisson_obs(n=10), **kw)


def test_an_unknown_series_and_a_backwards_clock_are_refused_naming_the_row():
    df = pl.DataFrame({"series": ["a", "b", "z"], "t": [1.0, 2.0, 3.0], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(Exception, match="row 2.*'z'|row 2.*\"z\""):
        run(df, names=["a", "b"])
    back = pl.DataFrame({"series": ["a", "b", "a"], "t": [1.0, 5.0, 2.0], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(Exception, match="row 2.*clock order"):
        run(back, names=["a", "b"])


# --- task 105: the rules of `po.stream` ---------------------------------------


def test_the_same_kind_of_frame_comes_back():
    df = poisson_obs(n=80)
    kw = dict(series="series", names=NAMES, clock="t", value="v")
    eager = stream.refresh_time(df, **kw)
    lazy = stream.refresh_time(df.lazy(), **kw)
    assert isinstance(eager, pl.DataFrame) and isinstance(lazy, pl.LazyFrame)
    assert eager.equals(lazy.collect())


@pytest.mark.parametrize("split", [1, 37, 150, 239])
def test_two_runs_through_a_saved_state_give_the_one_runs_grid(tmp_path, split):
    """Rule 5, a stateful transform resumes: the state holds each group's
    grid part-way through an interval and its last clock, so a stream fed in
    two runs is the stream fed in one, however it is split."""
    a = poisson_obs(n=120, seed=6).with_columns(g=pl.lit("x"))
    b = poisson_obs(n=120, seed=7).with_columns(g=pl.lit("y"))
    df = pl.concat([a, b]).sort("t")
    kw = dict(series="series", names=NAMES, clock="t", value="v", group="g")
    want = stream.refresh_time(df, **kw)
    state = tmp_path / "grid.state"
    first = stream.refresh_time(df.head(split).lazy(), save_state=state, chunk_size=7, **kw)
    assert not state.exists(), "a plan writes nothing until it runs"
    first = first.collect()
    second = stream.refresh_time(df.slice(split), load_state=state, save_state=state, **kw)
    assert pl.concat([first, second]).equals(want)
    # And the state after the second run is the one run's, byte for byte.
    whole = tmp_path / "whole.state"
    stream.refresh_time(df, save_state=whole, **kw)
    assert state.read_bytes() == whole.read_bytes()


# `head(0)` is left out: whether polars runs the source for it at all is
# polars' choice; a limit of 0 is `refresh.rs`'s own test.
@pytest.mark.parametrize("n", [1, 3, 17])
def test_a_slice_saves_the_state_after_the_tick_behind_its_last_point(tmp_path, n):
    """Under `.head(n)`, the input is read up to the tick that completed the
    n-th point and no further: the state saved is the same whatever the chunk
    size, and a run resumed on the input after that tick goes on with point
    n + 1, so the two runs give the one run's grid. It saved the whole input's
    state once, and the points after the slice could then never be produced
    (the user's call, 2026-09-28: the bank's rule, the state after the input
    behind the rows returned)."""
    df = poisson_obs(n=200)  # sorted, continuous clock: one tick per clock value
    kw = dict(series="series", names=NAMES, clock="t", value="v")
    whole = stream.refresh_time(df, **kw)
    states = []
    for size in (1, 7, 100_000):
        path = tmp_path / f"head{size}.state"
        head = (
            stream.refresh_time(df.lazy(), save_state=path, chunk_size=size, **kw).head(n).collect()
        )
        assert head.equals(whole.head(n)), size
        states.append(path.read_bytes())
    assert states[0] == states[1] == states[2]
    # The input behind the slice ends at the tick of its last point.
    rest = df.filter(pl.col("t") > head["time_refresh"][-1])
    resumed = stream.refresh_time(rest, load_state=tmp_path / "head1.state", **kw)
    assert pl.concat([head, resumed]).equals(whole)


def test_a_state_is_read_when_the_plan_is_built(tmp_path):
    df = poisson_obs(n=200)
    kw = dict(series="series", names=NAMES, clock="t", value="v")
    state = tmp_path / "grid.state"
    stream.refresh_time(df.head(100), save_state=state, **kw)
    plan = stream.refresh_time(df.slice(100).lazy(), load_state=state, **kw)
    first = plan.collect()
    stream.refresh_time(df.head(50), save_state=state, **kw)
    assert plan.collect().equals(first)


def test_what_a_state_refuses(tmp_path):
    df = poisson_obs(n=100)
    kw = dict(series="series", clock="t", value="v")
    state = tmp_path / "grid.state"
    stream.refresh_time(df.head(60), names=NAMES, save_state=state, **kw)
    with pytest.raises(ValueError, match="saved with names"):
        stream.refresh_time(df, names=NAMES[:2], load_state=state, **kw)
    with pytest.raises(ValueError, match="pairs"):
        stream.refresh_time(df, names=NAMES, pairs=True, load_state=state, **kw)
    # The rows the state has learned, again: a step back, refused by row.
    with pytest.raises(Exception, match="row 0.*clock order"):
        stream.refresh_time(df, names=NAMES, load_state=state, **kw)
    with pytest.raises(FileNotFoundError):
        stream.refresh_time(df, names=NAMES, load_state=tmp_path / "none.state", **kw)
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        stream.refresh_time(df, names=NAMES, save_state=tmp_path / "no" / "x.state", **kw)
    (tmp_path / "junk.state").write_bytes(b"junk")
    with pytest.raises(ValueError, match="not a refresh_time state"):
        stream.refresh_time(df, names=NAMES, load_state=tmp_path / "junk.state", **kw)


def test_a_run_that_fails_writes_no_state(tmp_path):
    df = poisson_obs(n=100)
    back = df.with_columns(t=pl.when(pl.int_range(pl.len()) == 70).then(0.0).otherwise("t"))
    state = tmp_path / "grid.state"
    with pytest.raises(Exception, match="clock order"):
        stream.refresh_time(
            back, series="series", names=NAMES, clock="t", value="v", save_state=state
        )
    assert not state.exists()


def test_a_temporal_clock_is_ordered_to_the_nanosecond():
    """Task 120: read as a double of epoch nanoseconds, a step back of under
    256 ns was a tie at today's dates."""
    ns = 1_727_000_000_000_000_000
    df = pl.DataFrame(
        {
            "series": ["a", "b", "a"],
            "t": pl.Series([ns + 100, ns + 101, ns + 100 - 1]).cast(pl.Datetime("ns")),
            "v": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(Exception, match="row 2.*clock order"):
        stream.refresh_time(df, series="series", names=["a", "b"], clock="t", value="v")


def test_a_state_resumes_only_under_the_grouping_it_was_saved_with(tmp_path):
    """Review 2026-09-28: the file recorded names and pairs but not whether
    the sampler was grouped, so an ungrouped state loaded under group= and
    its rows were never checked against the saved clock."""
    df = poisson_obs(n=60).with_columns(g=pl.lit("x"))
    kw = dict(series="series", names=NAMES, clock="t", value="v")
    state = tmp_path / "grid.state"
    stream.refresh_time(df, save_state=state, **kw)
    with pytest.raises(ValueError, match="grouped"):
        stream.refresh_time(df, load_state=state, group="g", **kw)
    grouped = tmp_path / "grouped.state"
    stream.refresh_time(df, save_state=grouped, group="g", **kw)
    with pytest.raises(ValueError, match="grouped"):
        stream.refresh_time(df, load_state=grouped, **kw)


def test_a_state_of_the_other_kind_of_clock_is_refused_on_any_row(tmp_path):
    """Review 2026-09-28: the clock-kind check ran only for a group with a
    prior, so a temporal state resumed on a numeric clock passed when the
    chunk held new groups only, and the saved file then mixed the two."""
    df = poisson_obs(n=60).with_columns(g=pl.lit("x"))
    kw = dict(series="series", names=NAMES, clock="t", value="v", group="g")
    state = tmp_path / "grid.state"
    stream.refresh_time(df, save_state=state, **kw)
    temporal = df.with_columns(
        t=pl.from_epoch(pl.col("t").cast(pl.Int64), time_unit="s"), g=pl.lit("y")
    )
    with pytest.raises(Exception, match="temporal.*numeric|numeric.*temporal"):
        stream.refresh_time(temporal, load_state=state, **kw)


def _two_groups(n: int = 60) -> pl.DataFrame:
    """Blocks of a, b, c ticks, alternating between two groups, so each
    block completes a point of its group; ``i`` is the row."""
    rng = np.random.default_rng(31)
    return pl.DataFrame(
        {
            "series": [NAMES[i % 3] for i in range(n)],
            "t": [float(i) for i in range(n)],
            "v": rng.standard_normal(n),
            "k": [(i // 3) % 2 for i in range(n)],
            "i": list(range(n)),
        }
    )


_GROUP_DTYPES = {
    "Int64": lambda k: k.cast(pl.Int64),
    "String": lambda k: k.cast(pl.String).replace_strict({"0": "x", "1": "y"}),
    "Date": lambda k: pl.date(2024, 1, 1) + pl.duration(days=k),
    "Datetime(us)": lambda k: pl.datetime(2024, 1, 1, time_unit="us") + pl.duration(seconds=k),
    "Datetime(ms, tz)": lambda k: (
        pl.datetime(2024, 1, 1, time_unit="ms", time_zone="Europe/Amsterdam")
        + pl.duration(milliseconds=k)
    ),
    "Time": lambda k: pl.when(k == 0).then(pl.time(9)).otherwise(pl.time(10, 30)),
    "Boolean": lambda k: k == 1,
    "Struct": lambda k: pl.struct(a=k, b=pl.lit("q")),
}


@pytest.mark.parametrize("dtype", list(_GROUP_DTYPES), ids=list(_GROUP_DTYPES))
def test_the_group_column_is_the_inputs_at_the_completing_ticks(dtype: str) -> None:
    """Task 160, PA4: the keys were held as text and cast back, so a
    Datetime(us), Time or Struct group came back null on every row, and a
    Boolean or tz-aware Datetime group failed the cast. The column is the
    input's own, taken at each point's completing tick, as ``keep`` is."""
    df = _two_groups().with_columns(g=_GROUP_DTYPES[dtype](pl.col("k")))
    lazy = stream.refresh_time(
        df.lazy(), series="series", names=NAMES, clock="t", value="v", group="g", keep=["i"]
    )
    out = lazy.collect()
    assert out.height == 20, "each block of three completes one point"
    assert out.schema["g"] == df.schema["g"] == lazy.collect_schema()["g"]
    want = df["g"].gather(out["i"])
    assert out["g"].null_count() == 0
    assert out["g"].equals(want), (out["g"], want)


_CLOCK_DTYPES = {
    "Float64": lambda t: t.cast(pl.Float64),
    "Int64": lambda t: t.cast(pl.Int64) * 7 + 3,
    "Datetime(ns)": lambda t: (pl.lit(1_700_000_000_000_000_001) + t.cast(pl.Int64) * 1_000).cast(
        pl.Datetime("ns")
    ),
    "Datetime(us, tz)": lambda t: (pl.lit(1_700_000_000_000_001) + t.cast(pl.Int64) * 1_000).cast(
        pl.Datetime("us", "UTC")
    ),
    "Date": lambda t: pl.date(2024, 1, 1) + pl.duration(days=t.cast(pl.Int64)),
}


@pytest.mark.parametrize("dtype", list(_CLOCK_DTYPES), ids=list(_CLOCK_DTYPES))
def test_time_refresh_is_the_completing_ticks_clock_in_its_own_dtype(dtype: str) -> None:
    """Task 160, PA6: ``time_refresh`` was the clock's physical integer as a
    Float64, so a nanosecond clock's 1700000000000002001 came out as
    ...2048, and a Datetime as a bare number. It is the completing tick's
    clock, exactly, in the clock column's dtype."""
    df = _two_groups().with_columns(t=_CLOCK_DTYPES[dtype](pl.col("t")))
    lazy = stream.refresh_time(
        df.lazy(), series="series", names=NAMES, clock="t", value="v", keep=["i"]
    )
    out = lazy.collect()
    assert out.height > 10
    assert out.schema["time_refresh"] == df.schema["t"] == lazy.collect_schema()["time_refresh"]
    assert out["time_refresh"].equals(df["t"].gather(out["i"]).alias("time_refresh"))


# --- task 278 (b26): names, pairs, group and keep together ---------------------
#
# Every comparison here is to the bit: a grid point is a tick's own clock and
# values, the counts are integers, and ``retained_fraction`` is the number of
# series over an integer sum, which no reordering of the series or the chunks
# changes. Series are named x, y, z so that they are never confused with the
# pairs columns' generic ``a`` and ``b``.

XYZ = ["x", "y", "z"]


def xyz_obs(n=200, seed=0, rates=(1.0, 0.4, 0.15)):
    """``poisson_obs`` on x, y, z, with the row ``i`` to read the completing
    tick by."""
    return (
        poisson_obs(names=XYZ, rates=rates, n=n, seed=seed)
        .with_row_index("i")
        .with_columns(pl.col("i").cast(pl.Int64))
    )


def xyz(df, **kw):
    kw.setdefault("names", XYZ)
    return run(df, **kw)


def oriented(out, names, order):
    """A pairs frame run under ``names``, relabelled to the orientation that
    ``order`` gives: a pair whose series come the other way round in
    ``order`` has its label reversed and its ``a``/``b`` columns swapped.
    Sorted by tick and pair, as two runs need not emit one tick's pairs in
    the same order."""
    rank = {s: k for k, s in enumerate(order)}
    reversed_labels = [f"{a}|{b}" for a in names for b in names if rank[a] > rank[b] and a != b]
    out = out.with_columns(_swap=pl.col("pair").is_in(reversed_labels))

    def pick(a, b):
        return pl.when(pl.col("_swap")).then(pl.col(b)).otherwise(pl.col(a))

    return (
        out.with_columns(
            pair=pl.when(pl.col("_swap"))
            .then(pl.col("pair").str.split("|").list.reverse().list.join("|"))
            .otherwise(pl.col("pair")),
            a_value=pick("a_value", "b_value"),
            b_value=pick("b_value", "a_value"),
            n_obs_a=pick("n_obs_a", "n_obs_b"),
            n_obs_b=pick("n_obs_b", "n_obs_a"),
        )
        .drop("_swap")
        .sort("i", "pair")
    )


def multi_pair_ticks(out):
    """The ticks (by ``i``) that completed more than one pair's point."""
    return out.group_by("i").len().filter(pl.col("len") > 1)["i"].sort()


def test_permuting_names_permutes_the_columns_and_nothing_else():
    """``names`` sets the output's column order, not the grid: a point is
    where the set of series completes, whatever order the set is listed in."""
    df = xyz_obs()
    want = xyz(df, keep=["i"])
    for order in (["z", "x", "y"], ["y", "z", "x"], ["z", "y", "x"]):
        got = xyz(df, names=order, keep=["i"])
        assert got.columns == [
            "time_refresh",
            *(f"{s}_value" for s in order),
            *(f"n_obs_{s}" for s in order),
            "retained_fraction",
            "i",
        ]
        assert got.select(want.columns).equals(want), order
    assert want.height > 20, "the grid has points to compare"


def test_permuting_names_reorients_the_pairs_and_nothing_else():
    """Under ``pairs=True`` the label is in ``names`` order, so a permutation
    that reverses two series reverses their label and swaps their ``a`` and
    ``b`` columns; the points are the same."""
    df = xyz_obs()
    want = oriented(xyz(df, pairs=True, keep=["i"]), XYZ, XYZ)
    for order in (["z", "x", "y"], ["z", "y", "x"]):
        raw = xyz(df, names=order, pairs=True, keep=["i"])
        # Activation: some labels did come out the other way round.
        assert set(raw["pair"]) - set(want["pair"]), order
        assert oriented(raw, order, XYZ).equals(want), order


def test_pairs_with_two_names_are_the_joint_grid():
    """With two series there is one pair, and its grid is the joint grid,
    relabelled: ``pair`` added, ``<s>_value`` and ``n_obs_<s>`` named for
    ``a`` and ``b``."""
    df = xyz_obs().filter(pl.col("series").is_in(["x", "z"]))
    joint = xyz(df, names=["x", "z"], keep=["i"])
    pairs = xyz(df, names=["x", "z"], pairs=True, keep=["i"])
    assert joint.height > 20
    want = joint.select(
        pl.lit("x|z").alias("pair"),
        "time_refresh",
        pl.col("x_value").alias("a_value"),
        pl.col("z_value").alias("b_value"),
        pl.col("n_obs_x").alias("n_obs_a"),
        pl.col("n_obs_z").alias("n_obs_b"),
        "retained_fraction",
        "i",
    )
    assert pairs.equals(want)


def test_keep_columns_under_pairs_come_last_at_the_completing_tick():
    """``keep`` under ``pairs=True`` is what it is on the joint grid: the
    column after the pairs columns, at its value on the tick that completed
    the pair's point. The point is the longhand loop's."""
    df = xyz_obs(n=120).with_columns(tag=pl.format("r{}", pl.col("i")), w=pl.col("v") * 2.0)
    out = xyz(df, pairs=True, keep=["tag", "w"])
    assert out.columns == [
        "pair",
        "time_refresh",
        "a_value",
        "b_value",
        "n_obs_a",
        "n_obs_b",
        "retained_fraction",
        "tag",
        "w",
    ]
    assert out.schema["tag"] == pl.String and out.schema["w"] == pl.Float64
    # The completing tick, found by its clock (one tick per clock value).
    tick = out.join(
        df.select("t", "i", tag_in="tag", w_in="w"), left_on="time_refresh", right_on="t"
    )
    assert tick.height == out.height
    assert tick["tag"].equals(tick["tag_in"].alias("tag"))
    assert tick["w"].equals(tick["w_in"].alias("w"))
    # And the pairs columns are the longhand loop's.
    want = pl.DataFrame(oracle(df, XYZ, pairs=True))
    for p in ("x|y", "x|z", "y|z"):
        a, b = p.split("|")
        mine = out.filter(pl.col("pair") == p).drop("tag", "w", "pair")
        theirs = want.filter(pl.col("pair") == p).select(
            "time_refresh",
            pl.col(f"{a}_value").alias("a_value"),
            pl.col(f"{b}_value").alias("b_value"),
            pl.col(f"n_obs_{a}").alias("n_obs_a"),
            pl.col(f"n_obs_{b}").alias("n_obs_b"),
            "retained_fraction",
        )
        assert mine.equals(theirs), p


def two_groups_xyz(n=120):
    a = xyz_obs(n=n, seed=8).with_columns(g=pl.lit("p"))
    b = xyz_obs(n=n, seed=9).with_columns(g=pl.lit("q"))
    return (
        pl.concat([a, b])
        .sort("t")
        .drop("i")
        .with_row_index("i")
        .with_columns(pl.col("i").cast(pl.Int64))
    )


def test_pairs_under_a_group_are_each_groups_own_pairs():
    """``group`` x ``pairs``: each group runs its own pairs, so a group's rows
    are the pairs grid of its part of the input alone, and the group column
    comes first."""
    df = two_groups_xyz()
    out = xyz(df, pairs=True, group="g", keep=["i"])
    assert out.columns[:2] == ["g", "pair"]
    # Activation: the groups interleave, so each group's grid is read across
    # the other's ticks.
    assert out["g"].head(20).n_unique() == 2
    for key in ("p", "q"):
        part = df.filter(pl.col("g") == key)
        want = xyz(part, pairs=True, keep=["i"])
        got = out.filter(pl.col("g") == key).drop("g")
        assert want.height > 30 and got.equals(want), key


def _pairs_split_case():
    """Two interleaved groups under pairs: 600 rows, 100 ticks a series a
    group."""
    df = two_groups_xyz(n=100)
    kw = dict(series="series", names=XYZ, clock="t", value="v", group="g", pairs=True, keep=["i"])
    return df, kw


@pytest.mark.parametrize("chunks", [1, 7, 600])
def test_pairs_are_the_same_from_one_chunk_or_many(chunks):
    """Pairs are chunk-invariant as the joint grid is, including where one
    tick completes several pairs and, at one row a chunk, every such tick
    ends its chunk."""
    df, kw = _pairs_split_case()
    assert df.height == 600
    want = stream.refresh_time(df, chunk_size=df.height, **kw)
    multi = multi_pair_ticks(want)
    assert multi.len() > 10, "ticks completing several pairs at once"
    size = -(-df.height // chunks)  # 600 rows in 1, 7 or 600 chunks
    got = stream.refresh_time(df.lazy(), chunk_size=size, **kw).collect()
    assert got.equals(want)


@pytest.mark.parametrize("split", [1, 50, 237, 400, "after a multi-pair tick"])
def test_pairs_through_a_saved_state_are_the_one_runs_pairs(tmp_path, split):
    """A pairs state holds each group's every pair part-way through its
    interval, so two runs give the one run's pairs and end in its state."""
    df, kw = _pairs_split_case()
    want = stream.refresh_time(df, **kw)
    if isinstance(split, str):
        # The first run ends on a tick that completed several pairs at once.
        split = multi_pair_ticks(want)[0] + 1
    state = tmp_path / "pairs.state"
    first = stream.refresh_time(df.head(split).lazy(), save_state=state, chunk_size=7, **kw)
    first = first.collect()
    second = stream.refresh_time(df.slice(split), load_state=state, save_state=state, **kw)
    assert pl.concat([first, second]).equals(want)
    whole = tmp_path / "whole.state"
    stream.refresh_time(df, save_state=whole, **kw)
    assert state.read_bytes() == whole.read_bytes()


def test_a_slice_ending_among_one_ticks_pairs_saves_the_state_after_that_tick(tmp_path):
    """The docstring's warning, held: one tick can complete several pairs'
    points, and a ``head(n)`` that ends among them reads up to that tick and
    saves the state after it, whatever the chunk size. That state is the one
    a run over the input up to the tick saves, so the points the slice left
    out of that tick are returned by neither the slice nor a run resumed
    after the tick; every other point is."""
    df = xyz_obs()
    kw = dict(series="series", names=XYZ, clock="t", value="v", pairs=True, keep=["i"])
    whole = stream.refresh_time(df, **kw)
    tick = multi_pair_ticks(whole)[0]
    first_at_tick = whole.with_row_index("r").filter(pl.col("i") == tick)["r"]
    assert first_at_tick.len() >= 2, "the tick completes several pairs"
    n = first_at_tick[0] + 1  # the slice ends after the first of them
    states = []
    for size in (1, 7, 100_000):
        path = tmp_path / f"head{size}.state"
        plan = stream.refresh_time(df.lazy(), save_state=path, chunk_size=size, **kw)
        head = plan.head(n).collect()
        assert head.equals(whole.head(n)), size
        states.append(path.read_bytes())
    assert states[0] == states[1] == states[2]
    upto = tmp_path / "upto.state"
    stream.refresh_time(df.filter(pl.col("i") <= tick), save_state=upto, **kw)
    assert upto.read_bytes() == states[0]
    resumed = stream.refresh_time(
        df.filter(pl.col("i") > tick), load_state=tmp_path / "head1.state", **kw
    )
    lost = whole.slice(n, first_at_tick.len() - 1)
    assert lost["i"].to_list() == [tick] * (first_at_tick.len() - 1)
    assert pl.concat([head, lost, resumed]).equals(whole)


_GRID_ORDERS = [XYZ, ["z", "x", "y"], ["y", "x", "z"]]


@pytest.mark.extended(reason="a grid: 54 cases of up to 1.2 s, 15 s in all")
@pytest.mark.parametrize("seed", [11, 12, 13])
@pytest.mark.parametrize("chunks", [1, 7, 600])
@pytest.mark.parametrize("order", range(len(_GRID_ORDERS)))
@pytest.mark.parametrize("grouped", [False, True])
def test_pairs_grid_chunks_orders_groups_and_keep(seed, chunks, order, grouped):
    """Every combination of the options at once, on streams with a slow
    series: each group's pairs, chunked, under a permutation of ``names``
    and with a ``keep`` column, are the longhand loop's pairs in the
    canonical orientation."""
    names = _GRID_ORDERS[order]
    a = xyz_obs(n=200, seed=seed, rates=(2.0, 0.7, 0.1)).with_columns(g=pl.lit("p"))
    b = xyz_obs(n=200, seed=seed + 100, rates=(0.3, 1.5, 0.6)).with_columns(g=pl.lit("q"))
    df = pl.concat([a, b] if grouped else [a]).sort("t").drop("i")
    df = df.with_row_index("i").with_columns(pl.col("i").cast(pl.Int64))
    size = -(-df.height // chunks)
    kw = dict(series="series", names=names, clock="t", value="v", pairs=True, keep=["i"])
    if grouped:
        kw["group"] = "g"
    got = stream.refresh_time(df.lazy(), chunk_size=size, **kw).collect()
    assert multi_pair_ticks(got).len() > 0
    for key in ("p", "q") if grouped else (None,):
        part = df if key is None else df.filter(pl.col("g") == key)
        mine = got if key is None else got.filter(pl.col("g") == key).drop("g")
        rows = oracle(part, XYZ, pairs=True)
        tick_of = dict(zip(part["t"], part["i"], strict=True))
        want = pl.DataFrame(
            [
                {
                    "pair": r["pair"],
                    "time_refresh": r["time_refresh"],
                    "a_value": r[f"{sa}_value"],
                    "b_value": r[f"{sb}_value"],
                    "n_obs_a": r[f"n_obs_{sa}"],
                    "n_obs_b": r[f"n_obs_{sb}"],
                    "retained_fraction": r["retained_fraction"],
                    "i": tick_of[r["time_refresh"]],
                }
                for r in rows
                for sa, sb in [r["pair"].split("|")]
            ]
        ).sort("i", "pair")
        assert want.height > 30
        assert oriented(mine, names, XYZ).equals(want), (key, names)
