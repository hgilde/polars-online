"""Task 194: what the bank keeps between chunks, and the frames it reads back.

Review round 4 (docs/PLAN.md §18), the user's decisions of 2026-10-06:

- N22: a group or session column keeps the form it had at the first chunk --
  integer, float, text, boolean, or a temporal type with its unit and zone --
  because a key is its value's text, and a column of another form would
  start every group over, or cut every session, in silence.
- N18: the clock range in ``summary()``, ``groups()`` and ``closed_groups()``
  is in the clock column's own dtype, exactly, as ``emit_clocks`` is.
- N19, N20: ``closed_groups()`` counts as ``UInt64``, and prefixes every
  column of its ``rcov`` block.
- N21: ``group=`` takes one key or a list of them, ``None`` among them for
  the null group, and integer keys list in numeric order.
- S2: without a cadence, ``coef`` is written on each group's last accepted
  row of a chunk.
- S3: ``predict`` under an embargo releases nothing.
- N9, N8: ``rows_fed()``; ``t_stat`` and ``pair_t_stat``.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import polars_online as po

pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")


def ridge(**kw):
    kw.setdefault("half_life", 10.0)
    kw.setdefault("min_weight", 1.0)
    return po.spec.ewridge("m", targets=["y"], features=["x"], **kw)


def keyed(n=40, seed=0):
    """Two groups, interleaved, keyed by an Int64 column ``g``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return pl.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "g": [i % 2 for i in range(n)],
            "x": x,
            "y": 2.0 * x + 0.1 * rng.standard_normal(n),
        }
    )


def weight_sums(out: pl.DataFrame) -> list[float | None]:
    return out["m"].struct.field("weight_sum").to_list()


# --- N22: a key column keeps its form -------------------------------------------


@pytest.mark.parametrize(
    ("cast", "now"),
    [(pl.Float64, "f64"), (pl.String, "str"), (pl.Boolean, "bool")],
)
def test_a_group_column_of_another_form_is_refused_by_name(cast, now):
    """Int64 keys then Float64 ones were the groups '1' and '1.0', two
    streams, the second cold (review round 4, PC1, PA7)."""
    df = keyed()
    bank = po.ModelBank([ridge(group="g")])
    bank.fit_predict(df.head(20))
    before = bank.save_bytes()
    with pytest.raises(ValueError, match=rf'group column "g" was i64 and is now {now}'):
        bank.fit_predict(df.slice(20).with_columns(pl.col("g").cast(cast)))
    assert bank.save_bytes() == before, "a refused chunk moves nothing"


def test_a_session_column_of_another_form_is_refused_by_name():
    """An Int64 session then a Float64 one read '1' then '1.0': a session
    change on every group, and under ``session_gap = "reset"`` a cold
    model (PC-battery2)."""
    df = keyed().with_columns(s=pl.lit(1))
    spec = ridge(session="s", session_gap="reset", clock="t", gap_cap=1e9)
    bank = po.ModelBank([spec])
    bank.fit_predict(df.head(20))
    with pytest.raises(ValueError, match=r'session column "s" was i32 and is now f64'):
        bank.fit_predict(df.slice(20).with_columns(pl.col("s").cast(pl.Float64)))


def test_the_form_is_kept_in_the_state_and_read_by_predict():
    df = keyed()
    bank = po.ModelBank([ridge(group="g")])
    bank.fit_predict(df.head(20))
    loaded = po.ModelBank.load_bytes(bank.save_bytes())
    floats = df.slice(20).with_columns(pl.col("g").cast(pl.Float64))
    with pytest.raises(ValueError, match=r'group column "g" was i64 and is now f64'):
        loaded.fit_predict(floats)
    with pytest.raises(ValueError, match=r'group column "g" was i64 and is now f64'):
        loaded.predict(floats)


@pytest.mark.parametrize(
    ("first", "then"),
    [
        (pl.Int64, pl.Int32),
        (pl.Int64, pl.UInt8),
        (pl.String, pl.Categorical),
    ],
)
def test_another_dtype_of_the_same_form_continues_the_stream(first, then):
    """What the refusal protects: the same keys in a dtype of the same form
    are the same groups, and the stream goes on as if fed in one chunk."""
    df = keyed()
    if first == pl.String:
        df = df.with_columns(pl.col("g").cast(pl.String))
    whole = weight_sums(po.ModelBank([ridge(group="g")]).fit_predict(df))
    bank = po.ModelBank([ridge(group="g")])
    head = weight_sums(bank.fit_predict(df.head(20)))
    tail = weight_sums(bank.fit_predict(df.slice(20).with_columns(pl.col("g").cast(then))))
    assert head + tail == whole


def test_a_null_typed_column_carries_no_form():
    """A column of nulls with no type of its own, ``pl.lit(None)``, is the
    null key in any form: neither recorded nor refused."""
    df = keyed()
    bank = po.ModelBank([ridge(group="g")])
    bank.fit_predict(df.head(20).with_columns(g=pl.lit(None)))
    bank.fit_predict(df.slice(20, 10))
    with pytest.raises(ValueError, match=r'group column "g" was i64 and is now f64'):
        bank.fit_predict(df.slice(30).with_columns(pl.col("g").cast(pl.Float64)))


# --- N18: the clock range in the clock's own dtype ------------------------------


def clocked(dtype: pl.DataType, n: int = 12) -> pl.DataFrame:
    """Two groups and a clock of ``dtype``, its values off a double's grid:
    nanoseconds at 2024 a double of seconds resolves only to ~240 ns."""
    base = 1_704_187_800_126_456_001  # ns, 2024-01-02, odd to the nanosecond
    if dtype == pl.Date:
        t = pl.Series("t", [date(2024, 1, 2) + timedelta(days=i) for i in range(n)])
    elif dtype == pl.Int64:
        t = pl.Series("t", [1_000_000_007 + 3 * i for i in range(n)], dtype=pl.Int64)
    elif dtype == pl.Float64:
        t = pl.Series("t", [1.0e9 + 0.1 * i for i in range(n)], dtype=pl.Float64)
    else:
        unit = dtype.time_unit  # type: ignore[attr-defined]
        per = {"ns": 1, "us": 1_000, "ms": 1_000_000}[unit]
        raw = pl.Series("t", [(base + 1_234_567 * i) // per for i in range(n)], dtype=pl.Int64)
        t = raw.cast(pl.Datetime(unit)).dt.replace_time_zone(dtype.time_zone)  # type: ignore[attr-defined]
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n)
    return pl.DataFrame(
        {"g": [i // (n // 2) for i in range(n)], "x": x, "y": 2.0 * x}
    ).with_columns(t)


def clock_spec(df: pl.DataFrame, **kw):
    temporal = df["t"].dtype.is_temporal()
    return ridge(
        clock="t",
        group="g",
        half_life="30d" if temporal else 30.0,
        gap_cap="1000d" if temporal else 1e12,
        emit_clocks=True,
        **kw,
    )


DTYPES = [
    pl.Datetime("us", "UTC"),
    pl.Datetime("ns"),
    pl.Date(),
    pl.Int64(),
    pl.Float64(),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_summary_and_groups_give_the_clock_in_its_own_dtype_exactly(dtype):
    df = clocked(dtype)
    bank = po.ModelBank([clock_spec(df)])
    out = bank.fit_predict(df)
    # The dtype `emit_clocks` gives it, the clock's own: an integer clock is
    # held as an integer (task 200).
    want = out["m"].struct.field("scored_clock").dtype
    assert want == dtype
    by = df.group_by("g", maintain_order=True).agg(
        lo=pl.col("t").min(), hi=pl.col("t").max(), last=pl.col("t").last()
    )
    summary = bank.summary()
    groups = bank.groups()
    for col in ("clock_min", "clock_max", "last_clock"):
        assert summary[col].dtype == want, col
    assert groups["last_clock"].dtype == want
    assert summary["clock_min"].to_list() == by["lo"].to_list()
    assert summary["clock_max"].to_list() == by["hi"].to_list()
    assert summary["last_clock"].to_list() == by["last"].to_list()
    assert groups["last_clock"].to_list() == by["last"].to_list()


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_closed_groups_give_the_clock_in_its_own_dtype_exactly(dtype):
    df = clocked(dtype)
    bank = po.ModelBank([clock_spec(df, group_close="monotone")])
    bank.fit_predict(df)
    closed = bank.closed_groups()
    assert closed["clock_min"].dtype == dtype and closed["clock_max"].dtype == dtype
    first = df.filter(pl.col("g") == 0)["t"]
    assert closed["clock_min"].to_list() == [first.min()]
    assert closed["clock_max"].to_list() == [first.max()]


def test_a_saved_clock_range_keeps_its_nanoseconds():
    df = clocked(pl.Datetime("ns"))
    bank = po.ModelBank([clock_spec(df, group_close="monotone")])
    bank.fit_predict(df.head(9))
    loaded = po.ModelBank.load_bytes(bank.save_bytes())
    assert loaded.summary().equals(bank.summary())
    assert loaded.closed_groups().equals(bank.closed_groups())


def test_a_row_count_clock_reads_null_in_float64():
    df = keyed()
    bank = po.ModelBank([ridge(group="g")])
    bank.fit_predict(df)
    s = bank.summary()
    assert s["clock_min"].dtype == pl.Float64 and s["clock_min"].null_count() == s.height
    assert bank.groups()["last_clock"].dtype == pl.Float64


def test_specs_whose_clocks_differ_are_read_one_at_a_time():
    """One column holds one dtype: a bank whose specs read clocks of two
    dtypes is read spec by spec, and a spec with no clock does not count."""
    df = clocked(pl.Datetime("us")).with_columns(u=pl.col("t").dt.epoch("s").cast(pl.Float64))
    a = clock_spec(df)
    b = {**ridge(group="g", clock="u", gap_cap=1e12, half_life=1e6), "name": "b"}
    c = {**ridge(group="g"), "name": "c"}
    bank = po.ModelBank([a, b, c])
    bank.fit_predict(df)
    for read in (bank.summary, bank.groups):
        with pytest.raises(ValueError, match=r'spec "m".*datetime.*spec "b".*f64.*spec='):
            read()
        assert read("m")["last_clock"].dtype == pl.Datetime("us")
        assert read("b")["last_clock"].dtype == pl.Float64
    together = po.ModelBank([a, c])
    together.fit_predict(df)
    assert together.summary()["clock_min"].dtype == pl.Datetime("us")
    assert together.summary("c")["clock_min"].null_count() == 2


# --- N19, N20: the closed frame's counts and its rcov block ---------------------


def test_closed_counts_are_uint64_as_in_the_other_frames():
    df = keyed()
    bank = po.ModelBank([ridge(group="g", group_close="monotone")])
    bank.fit_predict(df.sort("g"))
    closed = bank.closed_groups()
    assert closed.height > 0
    assert closed.schema["rows_fed"] == pl.UInt64
    assert closed.schema["rows_learned"] == pl.UInt64
    summary = bank.summary()
    assert pl.concat([closed.select("rows_fed"), summary.select("rows_fed")]).height > 0


def test_every_rcov_column_carries_the_prefix():
    spec = po.spec.rcov(
        "r", features=["x0", "x1"], block_rows=10, group="g", group_close="monotone"
    )
    cols = po.ModelBank([spec]).closed_groups().columns
    block = [c for c in cols if c.startswith("rcov") or c == "rcorr"]
    assert block == [
        "rcov",
        "rcorr",
        "rcov_n",
        "rcov_kind",
        "rcov_bandwidth_used",
        "rcov_omega2",
        "rcov_iv_sparse",
        "rcov_iq",
        "rcov_psd_repaired",
    ]
    for old in ("bandwidth_used", "omega2", "iv_sparse", "iq", "psd_repaired"):
        assert old not in cols


# --- N21: reading a group, the null one included, in the column's order --------


def nulls_and_tens(n=44):
    """Keys 1..10 and a null every eleventh row (PA10)."""
    return pl.DataFrame(
        {
            "g": [None if i % 11 == 0 else (i % 11) for i in range(n)],
            "x": [0.1 * i for i in range(n)],
            "y": [0.2 * i + 1.0 for i in range(n)],
        }
    )


NUMERIC = [None, "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


def test_integer_keys_list_in_numeric_order_the_null_group_first():
    spec = po.spec.marginal("p", targets=["y"], features=["x"], group="g", half_life=50.0)
    bank = po.ModelBank([ridge(group="g"), spec])
    bank.fit_predict(nulls_and_tens())
    assert bank.groups("m")["group"].to_list() == NUMERIC
    assert bank.summary("m")["group"].to_list() == NUMERIC
    assert bank.describe("m")["group"].unique(maintain_order=True).to_list() == NUMERIC
    assert bank.last_row("m")["group"].to_list() == NUMERIC
    assert bank.coef("m")["group"].unique(maintain_order=True).to_list() == NUMERIC
    assert [g["group"] for g in bank.gram("m")] == NUMERIC
    assert bank.marginal("p")["group"].to_list() == NUMERIC
    assert list(bank.solve_failures()["m"]) == NUMERIC


def test_text_keys_list_in_text_order():
    df = nulls_and_tens().with_columns(pl.col("g").cast(pl.String))
    bank = po.ModelBank([ridge(group="g")])
    bank.fit_predict(df)
    assert bank.groups()["group"].to_list() == [
        None,
        "1",
        "10",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]


def test_the_null_group_is_read_alone_with_a_list():
    spec = po.spec.marginal("p", targets=["y"], features=["x"], group="g", half_life=50.0)
    bank = po.ModelBank([ridge(group="g"), spec])
    bank.fit_predict(nulls_and_tens())
    assert bank.summary("m", group=[None])["group"].to_list() == [None]
    assert bank.describe("m", group=[None])["group"].unique().to_list() == [None]
    assert bank.last_row("m", group=[None])["group"].to_list() == [None]
    assert bank.coef("m", group=[None])["group"].unique().to_list() == [None]
    assert [g["group"] for g in bank.gram("m", group=[None])] == [None]
    assert bank.marginal("p", group=[None])["group"].to_list() == [None]
    # A list of several, in the frame's order whatever the list's; one key
    # as a string, as before; `None` alone is every group.
    assert bank.summary("m", group=["10", None, "2"])["group"].to_list() == [None, "2", "10"]
    assert bank.summary("m", group="10")["group"].to_list() == ["10"]
    assert bank.summary("m", group=None).height == len(NUMERIC)
    assert bank.summary(group=[None, "nope"])["group"].to_list() == [None, None]


def test_a_group_list_takes_strings_and_none_only():
    bank = po.ModelBank([ridge(group="g")])
    bank.fit_predict(nulls_and_tens())
    with pytest.raises(TypeError, match=r"group.*str or None.*int"):
        bank.summary("m", group=[1])
    with pytest.raises(TypeError, match=r"group.*str or None.*int"):
        bank.coef("m", group=3)  # type: ignore[arg-type]


# --- S2: coef on the group's last accepted row ----------------------------------


def test_coef_rides_on_the_last_accepted_row_of_a_chunk():
    """A chunk whose last row of a group is skipped carried no coef for the
    group at all (review round 4, PB2)."""
    n = 40
    x = [float(i % 7) - 3.0 for i in range(n)]
    df = pl.DataFrame({"x": x, "y": [2.0 * v + 0.5 for v in x]})
    spec = ridge(half_life=20.0, min_weight=2.0, standardize=False, max_rows_between_solves=1)
    last_null = df.with_columns(
        x=pl.when(pl.int_range(pl.len()) == n - 1).then(None).otherwise(pl.col("x"))
    )

    def coef_rows(out):
        return out["m"].struct.field("coef").is_not_null().arg_true().to_list()

    assert coef_rows(po.ModelBank([spec]).fit_predict(df)) == [n - 1]
    assert coef_rows(po.ModelBank([spec]).fit_predict(last_null)) == [n - 2]
    bank = po.ModelBank([spec])
    assert coef_rows(bank.fit_predict(last_null)) == [n - 2]
    assert coef_rows(bank.fit_predict(df.slice(0, 5))) == [4]
    # The coefficients are the fit as of that row, as `coef()` reads it.
    fitted = po.ModelBank([spec])
    out = fitted.fit_predict(last_null)["m"].struct.field("coef")
    assert out[n - 2].to_list() == fitted.coef()["coef"].to_list()


def test_a_chunk_of_a_group_with_no_accepted_row_writes_no_coef():
    df = keyed()
    spec = ridge(group="g")
    bank = po.ModelBank([spec])
    bank.fit_predict(df.head(20))
    holes = df.slice(20, 6).with_columns(
        x=pl.when(pl.col("g") == 0).then(None).otherwise(pl.col("x"))
    )
    out = bank.fit_predict(holes).with_columns(c=pl.col("m").struct.field("coef"))
    assert out.filter(pl.col("g") == 0)["c"].null_count() == 3
    assert out.filter(pl.col("g") == 1)["c"].is_not_null().arg_true().to_list() == [2]


# --- S3: predict under an embargo releases nothing ------------------------------


def test_predict_under_an_embargo_releases_nothing():
    """The rows an embargo holds are learned by the next ``fit_predict``,
    never by ``predict``: every scored row sees the state as it stands
    (README, "Serving without learning"; review round 4, PB3 and TB6)."""
    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal(n)
    df = pl.DataFrame(
        {"t": np.arange(n, dtype=float), "x": x, "y": 2 * x + 0.1 * rng.standard_normal(n)}
    )
    spec = ridge(
        clock="t",
        gap_cap=1e9,
        half_life=50.0,
        min_weight=3.0,
        max_rows_between_solves=1,
        embargo=10.0,
        emit_clocks=True,
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df.head(100))  # rows 90..99 still held
    state = bank.save_bytes()
    last = bank.last_row()
    later = df.slice(100, 30)  # by t = 129 every held row's delay has passed
    scored = bank.predict(later)["m"].struct.unnest()
    assert bank.save_bytes() == state
    assert scored["learned_clock"].unique().to_list() == [89.0] == last["learned_clock"].to_list()
    assert scored["weight_sum"].n_unique() == 1
    learned = po.ModelBank.load_bytes(state).fit_predict(later.head(1))["m"].struct.unnest()
    assert learned["learned_clock"][0] == 90.0, "fit_predict releases what has matured"
    assert scored["weight_sum"][0] < learned["weight_sum"][0]


# --- N9, N8: the renamed count and statistics -----------------------------------


def test_rows_fed_is_the_frames_count_and_rows_seen_names_it():
    df = keyed()
    bank = po.ModelBank([ridge(group="g")])
    bank.fit_predict(df.head(30))
    bank.drop_groups(["0"])
    assert bank.rows_fed() == 30
    assert bank.summary()["rows_fed"].sum() < 30
    assert "rows_fed=30" in repr(bank)
    with pytest.raises(AttributeError, match=r"rows_seen was renamed rows_fed"):
        bank.rows_seen()


def test_the_t_statistic_is_t_stat_in_both_frames():
    df = keyed().sort("g")
    spec = po.spec.marginal(
        "p", targets=["y"], features=["x"], group="g", group_close="monotone", half_life=50.0
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    pairs = bank.marginal("p")
    assert "t_stat" in pairs.columns and "t" not in pairs.columns
    closed = bank.closed_groups()
    assert "pair_t_stat" in closed.columns and "pair_t" not in closed.columns
    # The closed row is `marginal()` read at the close.
    alone = po.ModelBank([spec])
    alone.fit_predict(df.filter(pl.col("g") == 0))
    assert closed["pair_t_stat"][0].to_list() == alone.marginal("p")["t_stat"].to_list()
