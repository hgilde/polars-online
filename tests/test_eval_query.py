"""Every `po.eval` call that reads a bank's output, over a query (task 380).

Given a ``LazyFrame``, the scores (``metrics``, ``sums``, ``window_metrics``,
``compare_specs``) read it in one pass a chunk at a time and return what they
return for the collected frame: every chunk reduced to the ten sums per key
that :func:`po.eval.sums` keeps, and the chunks merged as
:func:`po.eval.merge_sums` merges them. The tests of two specs keep their own
sums and carry each key's last rows from chunk to chunk. A call whose result
has a row per row (``unpack``, ``seqtest``, the per-row ``half_life`` tests)
returns a ``LazyFrame`` that collects to the frame's result. Held here:

- the numbers are the frame's: to the bit for ``sums``, for every column of
  the scores but ``ic``, and for the per-row tests, while the query is one
  chunk, and for ``seqtest`` at any chunk; within rounding across many
  chunks (1e-9 relative, the bound the merged sums are held to in
  ``test_eval_sums.py``, and 1e-12 for the per-row tests), at a level of 1e8
  too;
- a time-zoned clock's windows are counted from each key's first clock, as
  the frame's are, and a step back that would move them is refused;
- every refusal the frame gets, the query gets, with the same message,
  before the query runs;
- a query that runs a bank runs it once;
- the memory is flat in the query's length (extended: a child interpreter
  per call over one and four million rows).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

import child
import polars_online as po
from polars_online import _frame
from polars_version import needs_polars

TIER = "mixed"


def _out(n: int = 3000, seed: int = 0, offset: float = 0.0) -> tuple[pl.DataFrame, dict]:
    """A bank's output with every kind of spec the scores read: a grid of
    half-lives over two targets, a renamed target, a logistic fit, a
    Poisson fit (no sign to hit), and a group and a weight column."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal((2, n))
    y = offset + 2.0 * x0 - x1 + 0.5 * rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "x0": x0,
            "x1": x1,
            "y": y,
            "z": x1 + rng.standard_normal(n),
            "lab": (x0 + rng.standard_normal(n) > 0).astype(float),
            "count": rng.poisson(np.exp(0.3 + 0.5 * x0)).astype(float),
            "venue": rng.choice(["a", "b", "c"], n),
            "w": rng.uniform(0.0, 2.0, n),
        }
    )
    specs = {
        "grid": po.spec.ewridge(
            "grid", targets=["y", "z"], features=["x0", "x1"], half_life=[100.0, 1000.0],
            min_weight=5.0,
        ),
        "renamed": po.spec.ewridge(
            "renamed", targets=[po.target("y", name="ret")], features=["x0"], half_life=300.0
        ),
        "logistic": po.spec.sgd(
            "logistic", targets=["lab"], features=["x0", "x1"], loss="logistic", half_life=500.0
        ),
        "poisson": po.spec.sgd(
            "poisson", targets=["count"], features=["x0"], loss="poisson",
            half_life=float("inf"), learning_rate=0.01, min_weight=5.0,
        ),
    }  # fmt: skip
    return po.ModelBank(list(specs.values())).fit_predict(df), specs


@pytest.fixture(scope="module")
def fitted():
    return _out()


@pytest.fixture
def small_chunks(monkeypatch):
    """A query read 97 rows at a time, its sums merged every four chunks:
    every path of the merge in a few thousand rows."""
    monkeypatch.setattr(po.eval, "_chunk_rows", lambda: 97)
    monkeypatch.setattr(po.eval, "_MERGE_EVERY", 4)


def close(got: pl.DataFrame, want: pl.DataFrame, *, rel: float = 1e-9) -> None:
    assert got.columns == want.columns, (got.columns, want.columns)
    assert got.dtypes == want.dtypes, (got.dtypes, want.dtypes)
    assert got.height == want.height
    for c in got.columns:
        if got.schema[c] == pl.Float64:
            a, b = got[c].to_numpy(), want[c].to_numpy()
            assert (np.isnan(a) == np.isnan(b)).all(), c
            assert a[~np.isnan(a)] == pytest.approx(b[~np.isnan(b)], rel=rel), c
        else:
            assert got[c].to_list() == want[c].to_list(), c


CASES = [
    ("grid", {}),
    ("grid", {"group": "venue"}),
    ("renamed", {"group": ["venue"]}),
    ("logistic", {}),
    ("logistic", {"group": "venue"}),
    ("poisson", {}),
]
SUM_CASES = [
    *CASES,
    ("grid", {"weight": "w"}),
    ("grid", {"group": "venue", "weight": "w"}),
    ("logistic", {"weight": "w", "binary": True}),
]


class TestTheQueryGivesTheFramesNumbers:
    @pytest.mark.parametrize(("name", "kw"), CASES)
    def test_metrics_over_many_chunks(self, fitted, small_chunks, name, kw):
        out, specs = fitted
        close(
            po.eval.metrics(out.lazy(), specs[name], **kw), po.eval.metrics(out, specs[name], **kw)
        )

    @pytest.mark.parametrize(("name", "kw"), SUM_CASES)
    def test_sums_over_many_chunks(self, fitted, small_chunks, name, kw):
        out, specs = fitted
        close(po.eval.sums(out.lazy(), specs[name], **kw), po.eval.sums(out, specs[name], **kw))

    def test_the_small_chunks_are_read_and_merged(self, fitted, small_chunks, monkeypatch):
        """The fixture's activation: 3,000 rows in 97-row chunks are 31
        chunks, merged four parts at a time along the way (the first four
        chunks, then the running sums and three more) and once at the end."""
        out, specs = fitted
        merged = []
        real = po.eval._merge

        def counting(frames, keys, extra=()):
            merged.append(len(frames))
            return real(frames, keys, extra)

        monkeypatch.setattr(po.eval, "_merge", counting)
        po.eval.sums(out.lazy(), specs["grid"])
        assert merged == [4] * 10 + [1], merged

    @pytest.mark.parametrize(("name", "kw"), SUM_CASES)
    def test_sums_of_one_chunk_are_the_frames_to_the_bit(self, fitted, name, kw):
        out, specs = fitted
        assert po.eval.sums(out.lazy(), specs[name], **kw).equals(
            po.eval.sums(out, specs[name], **kw)
        )

    @pytest.mark.parametrize(("name", "kw"), CASES)
    def test_metrics_of_one_chunk_are_the_frames_to_the_bit_but_ic(self, fitted, name, kw):
        """``ic`` is Polars' ``corr`` over a frame and the co-moment over the
        root of the two moments from sums, which round apart."""
        out, specs = fitted
        got = po.eval.metrics(out.lazy(), specs[name], **kw)
        want = po.eval.metrics(out, specs[name], **kw)
        assert got.drop("ic").equals(want.drop("ic"))
        close(got.select("ic"), want.select("ic"), rel=1e-14)

    def test_a_name_reads_as_the_dict_does_on_a_plain_target(self, fitted, small_chunks):
        out, specs = fitted
        close(po.eval.metrics(out.lazy(), "grid"), po.eval.metrics(out, specs["grid"]))

    def test_min_samples_thins_a_key_as_the_frame_does(self, fitted, small_chunks):
        out, specs = fitted
        kw = {"group": "venue", "min_samples": 1200}
        got = po.eval.metrics(out.lazy(), specs["grid"], **kw)
        close(got, po.eval.metrics(out, specs["grid"], **kw))
        assert got["r2"].null_count() > 0, "a key thinned"
        assert got["n"].null_count() == 0

    @pytest.mark.parametrize("call", ["metrics", "sums"])
    def test_a_target_about_1e8_keeps_its_variance(self, small_chunks, call):
        """The reason the sums are centred: a target of unit spread about
        1e8, read 97 rows at a time, keeps its variance as the frame does."""
        out, _ = _out(n=4000, seed=4, offset=1e8)
        spec = po.spec.ewridge("plain", targets=["y"], features=["x0", "x1"], half_life=300.0)
        out = po.ModelBank([spec]).fit_predict(out.select("x0", "x1", "y"))
        fn = getattr(po.eval, call)
        close(fn(out.lazy(), spec), fn(out, spec))
        r2 = po.eval.metrics(out.lazy(), spec)["r2"][0]
        assert 0.7 < r2 < 1.0, r2

    def test_an_empty_query_gives_the_empty_frames_result(self, fitted):
        out, specs = fitted
        for call in (po.eval.metrics, po.eval.sums):
            got = call(out.lazy().head(0), specs["grid"], group="venue")
            assert got.equals(call(out.head(0), specs["grid"], group="venue"))
            assert got.height == 0 and "slot" in got.columns

    def test_a_query_that_runs_a_bank_runs_it_once(self, fitted):
        out, specs = fitted
        source = out.select("x0", "x1", "y", "z", "venue").lazy()
        query = source.online.fit_predict([specs["grid"]])
        before = _frame._source_runs()
        got = po.eval.metrics(query, specs["grid"], group="venue")
        assert _frame._source_runs() == before + 1
        close(got, po.eval.metrics(out, specs["grid"], group="venue"))


class TestUnpackIsLazyInLazyOut:
    @pytest.mark.parametrize("name", ["grid", "renamed", "logistic"])
    def test_it_collects_to_the_frames_long_form(self, fitted, name):
        out, specs = fitted
        got = po.eval.unpack(out.lazy(), specs[name])
        assert isinstance(got, pl.LazyFrame)
        assert got.collect().equals(po.eval.unpack(out, specs[name]))

    def test_by_name_with_targets(self, fitted):
        out, _ = fitted
        got = po.eval.unpack(out.lazy(), "grid", targets=["y", "z"]).collect()
        assert got.equals(po.eval.unpack(out, "grid", targets=["y", "z"]))


def _boom(s: pl.Series) -> pl.Series:
    raise AssertionError("the query ran")


def _refusals(out: pl.DataFrame, specs: dict) -> list[tuple[str, tuple, dict, pl.DataFrame]]:
    """Each refusal a frame gets: the call, its arguments, and the frame."""
    no_pred = out.select("y", pl.struct(pl.col("x0").alias("weight_sum")).alias("cov"))
    no_y = out.drop("y")
    return [
        ("metrics", ("nope",), {}, out),
        ("metrics", ("x0",), {}, out),
        ("metrics", ("cov",), {}, no_pred),
        ("metrics", ("grid",), {"group": "nowhere"}, out),
        ("metrics", (specs["grid"],), {}, no_y),
        ("metrics", ("grid",), {}, no_y),
        ("metrics", ("grid",), {"targets": ["zzz"]}, out),
        ("metrics", (3,), {}, out),
        ("metrics", ({"model": {}},), {}, out),
        ("metrics", ("grid",), {"spec_name": "grid"}, out),
        ("metrics", ("grid",), {"spec": {**specs["grid"], "name": "other"}}, out),
        ("sums", ("grid",), {"weight": "nowhere"}, out),
        ("sums", ("grid",), {"group": ["venue", "nowhere"]}, out),
        ("sums", ("renamed",), {}, out),
        ("sums", (specs["renamed"],), {}, out.drop("y")),
        ("unpack", ("nope",), {}, out),
        ("unpack", ("x0",), {}, out),
        ("unpack", (specs["grid"],), {}, no_y),
        ("unpack", ("grid",), {"targets": ["zzz"]}, out),
    ]


@pytest.mark.parametrize("case", range(19))
def test_a_query_is_refused_as_the_frame_is_before_it_runs(fitted, case):
    out, specs = fitted
    call, args, kw, frame = _refusals(out, specs)[case]
    fn = getattr(po.eval, call)
    with pytest.raises(Exception) as eager:
        fn(frame, *args, **kw)
    # A column whose computation fails if it runs: the refusal comes first.
    first = frame.columns[0]
    query = frame.lazy().with_columns(
        pl.col(first).map_batches(_boom, return_dtype=frame.schema[first])
    )
    with pytest.raises(type(eager.value)) as lazy:
        fn(query, *args, **kw)
    assert str(lazy.value) == str(eager.value)


def test_the_cases_are_all_counted(fitted):
    assert len(_refusals(*fitted)) == 19


@pytest.mark.parametrize("call", ["metrics", "sums", "unpack"])
def test_something_that_is_not_a_polars_frame_is_refused_by_name(fitted, call):
    out, _ = fitted
    with pytest.raises(TypeError, match=f"po.eval.{call}: the frame must be a Polars DataFrame "):
        getattr(po.eval, call)(out.to_dict(), "grid")


# --- the calls that read a whole frame: windows, several specs, two specs' tests ------


def _pair(n: int = 3000, seed: int = 7) -> pl.DataFrame:
    """Two nested specs over two targets, with a key that is sometimes
    null, and clocks of every kind: a float, an integer, a ``Datetime``, a
    time-zoned one and a ``Duration``."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal((2, n))
    seconds = np.cumsum(rng.exponential(9000.0, n)).astype(np.int64) * 1_000_000
    stamps = pl.Series(1_709_251_200_000_000 + seconds).cast(pl.Datetime("us"))
    df = pl.DataFrame(
        {
            "x0": x0,
            "x1": x1,
            "y": 2 * x0 - x1 + rng.standard_normal(n),
            "z": x1 + rng.standard_normal(n),
            "lab": (x0 + rng.standard_normal(n) > 0).astype(float),
            "g": rng.choice(["a", "b", None], n),
            "t": np.arange(n, dtype=float),
            "ti": np.arange(n),
            "ts": stamps,
        }
    ).with_columns(
        tz=pl.col("ts").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York"),
        dur=pl.col("ts") - pl.datetime(2024, 1, 1),
    )
    small = po.spec.ewridge(
        "small", targets=["y", "z"], features=["x0"], half_life=200.0, min_weight=5.0
    )
    big = po.spec.ewridge(
        "big", targets=["y", "z"], features=["x0", "x1"], half_life=200.0, min_weight=8.0
    )
    logistic = po.spec.sgd(
        "logistic", targets=["lab"], features=["x0", "x1"], loss="logistic", half_life=500.0
    )
    return po.ModelBank([small, big, logistic]).fit_predict(df)


@pytest.fixture(scope="module")
def pair():
    return _pair()


def _bits(frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` with each record column's fields as columns of their own."""
    for c, dtype in frame.schema.items():
        if isinstance(dtype, pl.Struct):
            fields = frame[c].struct.unnest().rename(lambda f, c=c: f"{c}.{f}")
            frame = frame.drop(c).hstack(fields)
    return frame


def same_bits(got: pl.DataFrame, want: pl.DataFrame) -> None:
    """Every value the same to the bit, a zero's sign and a NaN included."""
    got, want = _bits(got), _bits(want)
    assert got.columns == want.columns and got.dtypes == want.dtypes
    for c in got.columns:
        if got.schema[c] == pl.Float64:
            a, b = got[c].to_numpy(), want[c].to_numpy()
            assert (a.view(np.int64) == b.view(np.int64)).all(), c
        else:
            assert got[c].equals(want[c], check_names=True, null_equal=True), c


WINDOWS = [
    {"clock": "t", "every": 250.0},
    {"clock": "ti", "every": 300},
    {"clock": "ts", "every": "1d", "group": "g"},
    {"clock": "ts", "every": "1mo"},
    {"clock": "tz", "every": "90m"},
    {"clock": "tz", "every": "36h", "group": "g"},
    {"clock": "dur", "every": "36h"},
]


class TestWindowsOverAQuery:
    @pytest.mark.parametrize("kw", WINDOWS, ids=lambda kw: f"{kw['clock']}-{kw['every']}")
    def test_window_metrics_over_many_chunks(self, pair, small_chunks, kw):
        spec = "big"
        close(
            po.eval.window_metrics(pair.lazy(), spec, min_samples=1, **kw),
            po.eval.window_metrics(pair, spec, min_samples=1, **kw),
        )

    @pytest.mark.parametrize("kw", WINDOWS[:3], ids=["t", "ti", "ts"])
    def test_window_metrics_of_one_chunk_are_the_frames_to_the_bit_but_ic(self, pair, kw):
        got = po.eval.window_metrics(pair.lazy(), "big", **kw)
        want = po.eval.window_metrics(pair, "big", **kw)
        assert got.drop("ic").equals(want.drop("ic"))
        close(got.select("ic"), want.select("ic"), rel=1e-14)

    def test_a_binary_fit_in_windows(self, pair, small_chunks):
        kw = {"clock": "t", "every": 500.0, "group": "g", "binary": True}
        close(
            po.eval.window_metrics(pair.lazy(), "logistic", **kw),
            po.eval.window_metrics(pair, "logistic", **kw),
        )

    def test_a_time_zoned_clock_is_windowed_from_each_keys_first_clock(self, pair, small_chunks):
        """The case the anchors are for: Polars counts a time-zoned clock's
        90-minute windows from a key's first row, so the chunks windowed on
        their own put rows in other windows than the frame does (shown
        here), and windowed after a row at the key's first clock, in the
        frame's."""
        long = po.eval.unpack(pair, "big").filter(pl.col("tz").is_not_null())
        keys = ["slot", "target"]
        whole = po.eval._dynamic_windows(long, "tz", "90m", keys)["window_start"]
        apart = pl.concat(
            [
                po.eval._dynamic_windows(long.slice(o, 97), "tz", "90m", keys)
                for o in range(0, long.height, 97)
            ]
        )["window_start"]
        assert not whole.equals(apart), "the chunks on their own would be windowed alike"
        kw = {"clock": "tz", "every": "90m", "min_samples": 1}
        close(
            po.eval.window_metrics(pair.lazy(), "big", **kw),
            po.eval.window_metrics(pair, "big", **kw),
        )

    def test_a_step_back_is_refused_where_it_moves_the_windows(self, pair, small_chunks):
        later_first = pl.concat([pair.slice(500), pair.head(500)]).lazy()
        with pytest.raises(ValueError, match="steps back to .* Sort the query by the clock"):
            po.eval.window_metrics(later_first, "big", clock="tz", every="90m")
        # A calendar day starts at midnight whatever the first clock was.
        kw = {"clock": "tz", "every": "1d"}
        close(
            po.eval.window_metrics(later_first, "big", **kw),
            po.eval.window_metrics(later_first.collect(), "big", **kw),
        )


class TestSeveralSpecsOverAQuery:
    def test_compare_specs_over_many_chunks(self, pair, small_chunks):
        got = po.eval.compare_specs(pair.lazy(), ["small", "big"], group="g")
        close(got, po.eval.compare_specs(pair, ["small", "big"], group="g"))

    def test_no_specs_give_an_empty_frame(self, pair):
        assert po.eval.compare_specs(pair.lazy(), []).equals(po.eval.compare_specs(pair, []))

    def test_a_query_that_runs_a_bank_runs_it_once_for_every_spec(self, pair):
        specs = [
            po.spec.ewridge(name, targets=["y"], features=feats, half_life=200.0)
            for name, feats in (("one", ["x0"]), ("two", ["x0", "x1"]))
        ]
        query = pair.select("x0", "x1", "y").lazy().online.fit_predict(specs)
        before = _frame._source_runs()
        got = po.eval.compare_specs(query, specs)
        assert _frame._source_runs() == before + 1
        close(got, po.eval.compare_specs(query.collect(), specs))


TESTS = [
    ("diebold_mariano", {"a": "big", "b": "small"}),
    ("diebold_mariano", {"a": "big", "b": "small", "lags": 3, "group": "g"}),
    ("clark_west", {"big": "big", "small": "small", "lags": 2}),
    ("clark_west", {"big": "big", "small": "small", "lags": 1, "group": ["g"]}),
]


class TestTwoSpecsOverAQuery:
    @pytest.mark.parametrize("rows", [97, 1])
    @pytest.mark.parametrize(("call", "kw"), TESTS)
    def test_the_tests_over_every_row(self, pair, monkeypatch, rows, call, kw):
        """Within 1e-9 relative; measured 1e-12 at the worst, on a mean
        differential near zero read a row at a time."""
        monkeypatch.setattr(po.eval, "_chunk_rows", lambda: rows)
        monkeypatch.setattr(po.eval, "_MERGE_EVERY", 4)
        fn = getattr(po.eval, call)
        close(fn(pair.lazy(), **kw), fn(pair, **kw))

    @pytest.mark.parametrize(("call", "kw"), TESTS)
    def test_the_tests_per_row_exponentially_weighted(self, pair, small_chunks, call, kw):
        """Within 1e-12 relative; measured 3e-15 in 97-row chunks."""
        fn = getattr(po.eval, call)
        kw = {**kw, "half_life": 50.0}
        got = fn(pair.lazy(), **kw)
        assert isinstance(got, pl.LazyFrame)
        want = fn(pair, **kw)
        close(_bits(got.collect()), _bits(want), rel=1e-12)

    @pytest.mark.parametrize(("call", "kw"), TESTS)
    def test_one_chunk_per_row_is_the_frames_to_the_bit(self, pair, call, kw):
        fn = getattr(po.eval, call)
        kw = {**kw, "half_life": 50.0}
        same_bits(fn(pair.lazy(), **kw).collect(), fn(pair, **kw))

    @pytest.mark.parametrize("rows", [100_000, 97, 1])
    @pytest.mark.parametrize(
        "kw",
        [
            {"a": "big", "b": "small", "group": "g", "min_weight": 3.0},
            {"targets": ["y", "z"]},
            {"targets": ["y"], "group": ["g"], "name": "evidence"},
        ],
    )
    def test_seqtest_is_the_frames_to_the_bit_at_any_chunk(self, pair, monkeypatch, rows, kw):
        """The log wealth is a sum in row order, carried from chunk to chunk
        into the next chunk's first term, so even a row at a time adds the
        same terms in the same order as the frame."""
        monkeypatch.setattr(po.eval, "_chunk_rows", lambda: rows)
        got = po.eval.seqtest(pair.lazy(), **kw)
        assert isinstance(got, pl.LazyFrame)
        same_bits(got.collect(), po.eval.seqtest(pair, **kw))

    def test_a_query_of_the_result_reads_the_rows_it_asks_for(self, pair, small_chunks):
        """Polars hands the source the columns, the filter and the rows the
        query asks for, and applies none of them again."""
        kw = {"a": "big", "b": "small", "group": "g"}
        lazy, whole = po.eval.seqtest(pair.lazy(), **kw), po.eval.seqtest(pair, **kw)
        same_bits(lazy.select("seqtest").head(250).collect(), whole.select("seqtest").head(250))
        same_bits(lazy.filter(pl.col("g") == "a").collect(), whole.filter(pl.col("g") == "a"))
        same_bits(
            lazy.filter(pl.col("t") > 1000).select("t", "seqtest").collect(),
            whole.filter(pl.col("t") > 1000).select("t", "seqtest"),
        )


def _refusals2(out: pl.DataFrame) -> list[tuple[str, dict, pl.DataFrame]]:
    """Each refusal the frame gets from the calls that read a whole frame."""
    return [
        ("window_metrics", {"spec": "big", "clock": "nowhere", "every": 1.0}, out),
        ("window_metrics", {"spec": "big", "clock": "g", "every": 1.0}, out),
        ("window_metrics", {"spec": "big", "clock": "t", "every": -1.0}, out),
        ("window_metrics", {"spec": "big", "clock": "ti", "every": 2.5}, out),
        ("window_metrics", {"spec": "big", "clock": "ts", "every": 3.0}, out),
        ("window_metrics", {"spec": "nope", "clock": "t", "every": 1.0}, out),
        ("compare_specs", {"specs": ["big", "nope"]}, out),
        ("compare_specs", {"specs": ["big"], "group": "nowhere"}, out),
        ("diebold_mariano", {"a": "big", "b": "nope"}, out),
        ("diebold_mariano", {"a": "big", "b": "x0"}, out),
        ("diebold_mariano", {"a": "big", "b": "small", "lags": -1}, out),
        ("diebold_mariano", {"a": "big", "b": "small", "lags": 1.5}, out),
        ("diebold_mariano", {"a": "big", "b": "small", "half_life": 0.0}, out),
        ("diebold_mariano", {"a": "big", "b": "small", "half_life": 9.0, "name": "y"}, out),
        ("diebold_mariano", {"a": "big", "b": "small", "targets": ["w"]}, out),
        ("clark_west", {"big": "big", "small": "small", "group": "nowhere"}, out),
        ("clark_west", {"big": "big", "small": "small", "big_suffix": "@x"}, out),
        ("seqtest", {"a": "big"}, out),
        ("seqtest", {}, out),
        ("seqtest", {"targets": ["y"], "name": "y"}, out),
        ("seqtest", {"targets": ["nowhere"]}, out),
        ("seqtest", {"a": "big", "b": "small", "group": "nowhere"}, out),
    ]


@pytest.mark.parametrize("case", range(22))
def test_a_whole_frame_call_refuses_a_query_as_the_frame_before_it_runs(pair, case):
    call, kw, frame = _refusals2(pair)[case]
    fn = getattr(po.eval, call)
    with pytest.raises(Exception) as eager:
        fn(frame, **kw)
    first = frame.columns[0]
    query = frame.lazy().with_columns(
        pl.col(first).map_batches(_boom, return_dtype=frame.schema[first])
    )
    with pytest.raises(type(eager.value)) as lazy:
        fn(query, **kw)
    assert str(lazy.value) == str(eager.value)


def test_the_whole_frame_cases_are_all_counted(pair):
    assert len(_refusals2(pair)) == 22


# --- memory -----------------------------------------------------------------------

#: The child: one call over a scan of an output, and the most memory the
#: process held, sampled, over the baseline, in MiB, with the rows read. A
#: call that returns a query writes it to a file with ``sink_parquet``.
CHILD = textwrap.dedent(
    """
    import gc, json, os, sys, threading, time
    import polars as pl, polars_online as po, psutil

    proc = psutil.Process()
    case, src = sys.argv[1], sys.argv[2]
    lf = pl.scan_parquet(src)
    g = dict(group="g")
    two = dict(a="a", b="b", **g)
    calls = {
        "metrics": lambda: po.eval.metrics(lf, "bank", group="g"),
        "sums": lambda: po.eval.sums(lf, "bank", group="g"),
        "unpack": lambda: po.eval.unpack(lf, "bank"),
        "window_metrics": lambda: po.eval.window_metrics(lf, "a", clock="ts", every="1d", **g),
        "compare_specs": lambda: po.eval.compare_specs(lf, ["a", "b"], group="g"),
        "diebold_mariano": lambda: po.eval.diebold_mariano(lf, lags=5, **two),
        "clark_west": lambda: po.eval.clark_west(lf, big="a", small="b", lags=5, group="g"),
        "seqtest": lambda: po.eval.seqtest(lf, **two),
        "diebold_mariano_ew": lambda: po.eval.diebold_mariano(lf, lags=2, half_life=500.0, **two),
    }
    gc.collect()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.002)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    got = calls[case]()
    if isinstance(got, pl.LazyFrame):
        got.sink_parquet(src + ".out")
        rows = pl.scan_parquet(src + ".out").select(pl.len()).collect().item()
    else:
        rows = int(got["n"].sum())
    peak[0] = max(peak[0], proc.memory_info().rss)
    stop.set()
    sampler.join()
    print(json.dumps({"rows": rows, "mib": (peak[0] - base) / 2**20}))
    """
)

#: The two lengths of the scanned output, rows.
MEMORY_ROWS = (1_000_000, 4_000_000)

#: The slots of the scanned output of the calls that score one spec.
SLOTS = 20

#: Each call measured, and the rows it reads from an output of ``n`` rows:
#: a row per row and slot for the scores of one spec's 20 slots, two rows
#: per row for two specs, a row per row for a test of two specs.
MEMORY_CASES = {
    "metrics": SLOTS,
    "sums": SLOTS,
    "unpack": SLOTS,
    "window_metrics": 1,
    "compare_specs": 2,
    "diebold_mariano": 1,
    "clark_west": 1,
    "seqtest": 1,
    "diebold_mariano_ew": 1,
}

#: The most a call's peak may grow from the shorter output to the longer,
#: MiB. Measured on macOS (M4 Pro, py-polars 2.0.0, load 3-5, 2026-11-09)
#: as the most memory the process held at once (`lifetime_max_phys_footprint`),
#: grouped by a key of four values, with the allocator returning freed
#: pages at once, at 1, 4 and 16 million rows: `metrics` of the 20-slot
#: output 195, 200 and 203 MiB, where the same scores of the collected frame
#: held 2,193 at one million rows; `window_metrics` by the day 112, 112 and
#: 115; `compare_specs` 105, 111 and 115; `diebold_mariano` 102, 103 and
#: 105, where the collected frame's held 252 at one million; `clark_west`
#: 99, 102 and 105; `seqtest` written to a file 152, 170 and 174; and
#: `diebold_mariano` by the row, written to a file, 155, 167 and 172. The
#: 20-slot output is 168 bytes a row, so a reading that kept its rows would
#: grow by about 480 MiB from one length to the other; the two-spec output
#: is 44 bytes a row, about 125 MiB.
MEMORY_GROWTH_MIB = 60.0

#: The most the longer output's peak may be, MiB: a fifth of the 2,193 MiB
#: the collected 20-slot frame's scores held at a quarter of its length.
MEMORY_PEAK_MIB = 450.0


def _bank_output(path, rows: int) -> None:
    """``rows`` rows of a 20-slot output: a target ``y``, a key ``g`` and a
    record column ``bank`` of ``pred_y__s<i>``, written a million rows at a
    time."""
    rng = np.random.default_rng(0)
    parts = []
    for start in range(0, rows, 1_000_000):
        m = min(1_000_000, rows - start)
        y = rng.standard_normal(m)
        preds = [pl.Series(f"pred_y__s{i}", 0.5 * y + rng.standard_normal(m)) for i in range(SLOTS)]
        parts.append(
            pl.DataFrame({"g": rng.integers(0, 4, m), "y": y}).with_columns(
                pl.struct(preds).alias("bank")
            )
        )
    pl.concat(parts).write_parquet(path, row_group_size=100_000)


def _two_specs_output(path, rows: int) -> None:
    """``rows`` rows of two specs' output, ``a`` and ``b``, each with a
    prediction and a residual of ``y``, beside a key ``g`` and a clock
    ``ts`` a second a row, written a million rows at a time."""
    rng = np.random.default_rng(0)
    parts = []
    for start in range(0, rows, 1_000_000):
        m = min(1_000_000, rows - start)
        y = rng.standard_normal(m)
        pa, pb = 0.5 * y + rng.standard_normal(m), 0.4 * y + rng.standard_normal(m)
        second = (np.arange(start, start + m) + 1_700_000_000) * 1_000_000
        parts.append(
            pl.DataFrame({"g": rng.integers(0, 4, m), "y": y}).with_columns(
                ts=pl.Series(second).cast(pl.Datetime("us")),
                a=pl.struct(pl.Series("pred_y", pa), pl.Series("resid_y", y - pa)),
                b=pl.struct(pl.Series("pred_y", pb), pl.Series("resid_y", y - pb)),
            )
        )
    pl.concat(parts).write_parquet(path, row_group_size=100_000)


@pytest.mark.extended(reason="memory: two child interpreters over up to four million rows")
@pytest.mark.serial
@needs_polars(
    "2.0.0",
    "py-polars 1.x memory-maps the parquet file it scans, so the resident peak grows by the "
    "file's size",
)
@pytest.mark.parametrize("case", list(MEMORY_CASES))
def test_a_call_over_a_query_holds_memory_flat_in_its_length(tmp_path, case):
    peaks = []
    for rows in MEMORY_ROWS:
        src = tmp_path / f"out_{rows}.parquet"
        (_bank_output if MEMORY_CASES[case] == SLOTS else _two_specs_output)(src, rows)
        r = subprocess.run(
            [sys.executable, "-c", CHILD, case, str(src)],
            capture_output=True, text=True, encoding="utf-8", timeout=600,
            env=child.env(
                POLARS_MAX_THREADS="4", POLARS_ROW_GROUP_PREFETCH_SIZE="1",
                _RJEM_MALLOC_CONF="dirty_decay_ms:0,muzzy_decay_ms:0", MIMALLOC_PURGE_DELAY="0",
            ),
        )  # fmt: skip
        assert r.returncode == 0, r.stderr[-3000:]
        got = json.loads(r.stdout.strip().splitlines()[-1])
        # Every row was read: per slot, per spec, or per row of the output.
        assert got["rows"] == rows * MEMORY_CASES[case], got
        assert got["mib"] > 1.0, got
        peaks.append(got["mib"])
        src.unlink()
    assert peaks[1] - peaks[0] < MEMORY_GROWTH_MIB, (case, peaks)
    assert peaks[1] < MEMORY_PEAK_MIB, (case, peaks)


@pytest.mark.parametrize("call", ["diebold_mariano", "clark_west", "seqtest"])
def test_by_is_refused_by_name_on_every_test_of_two_specs(pair, call):
    """``diebold_mariano``'s refusal of ``by`` decorated a helper of its
    instead, so ``by=`` reached Python's own "unexpected keyword"."""
    names = {"clark_west": {"big": "big", "small": "small"}}.get(call, {"a": "big", "b": "small"})
    with pytest.raises(TypeError, match=f"po.eval.{call}: by was renamed group"):
        getattr(po.eval, call)(pair.lazy(), by="g", **names)
