"""`po.eval.metrics`, `sums` and `unpack` over a query (task 380).

Given a ``LazyFrame``, ``metrics`` and ``sums`` read it in one pass a chunk
at a time and return what they return for the collected frame: every chunk
reduced to the ten sums per key that :func:`po.eval.sums` keeps, and the
chunks merged as :func:`po.eval.merge_sums` merges them. ``unpack`` returns
a ``LazyFrame`` that collects to the frame's long form. Held here:

- the numbers are the frame's: to the bit for ``sums``, and for every
  column of ``metrics`` but ``ic``, while the query is one chunk; within
  the merge's rounding (1e-9 relative, the bound the merged sums are held
  to in ``test_eval_sums.py``) across many chunks, at a level of 1e8 too;
- every refusal the frame gets, the query gets, with the same message,
  before the query runs;
- a query that runs a bank runs it once;
- the memory is flat in the query's length (extended: a child interpreter
  over one and four million rows of a 20-slot output).
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


# --- memory -----------------------------------------------------------------------

#: The child: ``metrics`` of a scan of a 20-slot output, grouped, and the
#: most memory the process held, sampled, over the baseline, in MiB.
CHILD = textwrap.dedent(
    """
    import gc, json, sys, threading, time
    import polars as pl, polars_online as po, psutil

    proc = psutil.Process()
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
    scores = po.eval.metrics(pl.scan_parquet(sys.argv[1]), "bank", group="g")
    peak[0] = max(peak[0], proc.memory_info().rss)
    stop.set()
    sampler.join()
    print(json.dumps({"rows": int(scores["n"].sum()), "keys": scores.height,
                      "mib": (peak[0] - base) / 2**20}))
    """
)

#: The two lengths of the scanned output, rows.
MEMORY_ROWS = (1_000_000, 4_000_000)

#: The slots of the scanned output.
SLOTS = 20

#: The most the peak may grow from the shorter output to the longer, MiB.
#: Measured on macOS (M4 Pro, py-polars 2.0.0, load 3-5, 2026-11-09), as
#: the most memory the process held at once (`lifetime_max_phys_footprint`),
#: grouped by a key of four values, with the allocator returning freed pages
#: at once: 195, 200 and 203 MiB at 1, 4 and 16 million rows, and 103, 105
#: and 111 ungrouped. The same scores of the collected frame held 2,193 MiB
#: at one million rows. The output is 168 bytes a row, so a reading that
#: kept the rows would grow by about 480 MiB from one length to the other,
#: and its long form by 20 times that.
MEMORY_GROWTH_MIB = 60.0

#: The most the longer output's peak may be, MiB: a fifth of the 2,193 MiB
#: the collected frame's scores held at a quarter of its length.
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


@pytest.mark.extended(reason="memory: two child interpreters over up to four million rows")
@pytest.mark.serial
@needs_polars(
    "2.0.0",
    "py-polars 1.x memory-maps the parquet file it scans, so the resident peak grows by the "
    "file's size",
)
def test_metrics_of_a_query_hold_memory_flat_in_its_length(tmp_path):
    peaks = []
    for rows in MEMORY_ROWS:
        src = tmp_path / f"out_{rows}.parquet"
        _bank_output(src, rows)
        r = subprocess.run(
            [sys.executable, "-c", CHILD, str(src)],
            capture_output=True, text=True, encoding="utf-8", timeout=600,
            env=child.env(
                POLARS_MAX_THREADS="4", POLARS_ROW_GROUP_PREFETCH_SIZE="1",
                _RJEM_MALLOC_CONF="dirty_decay_ms:0,muzzy_decay_ms:0", MIMALLOC_PURGE_DELAY="0",
            ),
        )  # fmt: skip
        assert r.returncode == 0, r.stderr[-3000:]
        got = json.loads(r.stdout.strip().splitlines()[-1])
        # Every row of every slot was scored, per slot and key.
        assert got["rows"] == rows * SLOTS, got
        assert got["keys"] == SLOTS * 4, got
        assert got["mib"] > 1.0, got
        peaks.append(got["mib"])
        src.unlink()
    assert peaks[1] - peaks[0] < MEMORY_GROWTH_MIB, peaks
    assert peaks[1] < MEMORY_PEAK_MIB, peaks
