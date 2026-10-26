"""`threads`: the pool a bank, a plan or a Gram fit runs on (docs/PLAN.md
task 253).

``threads=None`` is the process's pool, which ``POLARS_ONLINE_MAX_THREADS``
sizes when it is first used; an integer is a pool of that many threads,
shared by every holder of the count. The claims are bits: the output and the
saved state are the same at every count and every chunking, on a bank of
several specs and groups whose wide specs split their own work across the
pool (``gram_threads="auto"``, ``shards="auto"``), and on specs that one
stream serves, whose runs are copied into the output as they end, beside
the learning of the next run. The count is not state: a
save carries none of it and a load takes its own. Every surface refuses a
count that is not an integer of 1 or more with the one message, and given a
count the bank never reads the variable. The Rust tests hold the pool
itself: the registry's sharing and release
(``crates/online-polars/src/pool.rs``) and that a bank of one thread runs
every task on it (``crates/online-polars/src/bank/threads_tests.rs``).

Every kind is held to the same bits (task 278, batch 31): each builder as
``MINIMAL`` builds it and at its fullest, on four groups with a clock and a
weight, at one thread, four and the process's pool -- every output field,
every table read from the bank, and the saved state. Run on CI under each
OS's pool, as part of the extended tier; ``kmeans`` is the essentials' case.
``huber`` and ``sgd``'s Huber loss run with the refit of their first rows on
(task 277). That the comparison compares work spread over several threads is
read from the CPU time each thread accrues.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import pytest

import child
import polars_online as po
from polars_online import _polars_online as native
from polars_online._bank import _threads_refusal
from polars_online._spec import _json
from test_model_registry import MINIMAL, fullest

TIER = "mixed"

REPO = Path(__file__).resolve().parent.parent

#: Wide enough that a row's Gram update splits across the pool
#: (`crates/online-core/src/ewcov/par.rs`, from 725 columns).
K_WIDE = 730
N = 600


def stream(seed: int = 5) -> pl.DataFrame:
    """Three groups of a wide design, a narrow target, a clock and weights
    with a zero among them."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, K_WIDE))
    beta = rng.standard_normal(K_WIDE) / np.sqrt(K_WIDE)
    y = X @ beta + 0.1 * rng.standard_normal(N)
    w = 0.5 + rng.random(N)
    w[::37] = 0.0
    df = pl.DataFrame({f"x{j}": X[:, j] for j in range(K_WIDE)})
    return df.with_columns(
        y=pl.Series(y),
        w=pl.Series(w),
        t=pl.Series(np.arange(float(N))),
        g=pl.Series(["a", "b", "c"] * (N // 3)),
    )


def specs() -> list[dict]:
    wide = [f"x{j}" for j in range(K_WIDE)]
    narrow = ["x0", "x1", "x2", "x3"]
    return [
        po.spec.ewridge(
            "wide",
            targets=["y"],
            features=wide,
            half_life=200.0,
            group="g",
            clock="t",
            gap_cap=10.0,
            weight="w",
            ridge=1e-3,
            min_weight=20.0,
            solve_every=50.0,
            max_rows_between_solves=100,
            gram_threads="auto",
        ),
        po.spec.ew_cov(
            "cov",
            features=wide,
            half_life=100.0,
            group="g",
            stats=["mean", "var"],
            min_weight=1.0,
            gram_threads="auto",
        ),
        po.spec.ewridge(
            "narrow",
            targets=["y"],
            features=narrow,
            half_life=[20.0, 80.0],
            group="g",
            gram_threads=4,
        ),
        po.spec.marginal(
            "screen",
            targets=["y"],
            features=wide[:64],
            half_life=100.0,
            shards="auto",
        ),
        po.spec.sgd("sgd", targets=["y"], features=narrow, half_life=50.0, group="g"),
    ]


def run(threads: int | None, chunks: int) -> tuple[pl.DataFrame, bytes]:
    bank = po.ModelBank(specs(), threads=threads)
    df = stream()
    size = -(-df.height // chunks)
    out = pl.concat([bank.fit_predict(c) for c in df.iter_slices(size)])
    return out.drop(df.columns), bank.save_bytes()


@pytest.fixture(scope="module")
def reference() -> tuple[pl.DataFrame, bytes]:
    return run(None, 1)


@pytest.mark.parametrize("chunks", [1, 7, N])
@pytest.mark.parametrize("threads", [None, 1, 2, 8])
def test_the_output_and_the_state_are_the_same_at_every_count(reference, threads, chunks):
    want, state = reference
    got, saved = run(threads, chunks)
    # `coef`, and the fields laid out beside it, are reported on each
    # chunk's last row of a group, so they move with the chunking; every
    # other field holds to the bit.
    if chunks == 1:
        assert got.equals(want, null_equal=True), threads
    else:
        for spec in want.columns:
            fields = [f for f in want[spec].struct.fields if "coef" not in f]
            assert (
                got[spec]
                .struct.unnest()
                .select(fields)
                .equals(want[spec].struct.unnest().select(fields), null_equal=True)
            ), (threads, chunks, spec)
    assert saved == state, (threads, chunks)


def test_the_wide_specs_ran():
    """The comparison above compares something: the wide fit predicts past
    its warm-up, and the wide statistics are there on every row."""
    out, _ = run(2, 1)
    assert out["wide"].struct.field("pred_y").drop_nulls().len() > N // 6
    assert out["cov"].struct.field("mean_x0").drop_nulls().len() > N // 2


#: Rows of the single-stream frame: about six runs of a 230-statistic
#: `ew_cov` (a thousand rows a run) in one chunk, three in each of two, and in
#: one chunk past the 4,096 rows from which a spec's output columns are
#: allocated and finished in parallel.
N_SINGLE = 6_000


def single_stream() -> tuple[pl.DataFrame, list[dict]]:
    """Specs that one stream serves -- no group -- so each run is copied into
    the output as it ends, overlapped with the learning of the next run: a
    wide `ew_cov` and a four-target `ewridge` grid, beside a spec of four
    groups, whose runs wait for the end of the chunk."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((N_SINGLE, 20))
    X[rng.random((N_SINGLE, 20)) < 0.01] = np.nan
    Y = X[:, :4] @ rng.standard_normal((4, 4)) + rng.standard_normal((N_SINGLE, 4))
    w = 0.5 + rng.random(N_SINGLE)
    w[::31] = 0.0
    df = (
        pl.DataFrame({f"x{j}": X[:, j] for j in range(20)})
        .with_columns(
            [pl.Series(f"y{j}", Y[:, j]) for j in range(4)]
            + [pl.Series("w", w), pl.Series("t", np.cumsum(rng.exponential(1.0, N_SINGLE)))]
            + [pl.Series("g", np.arange(N_SINGLE) % 4)]
        )
        .with_columns(pl.col(f"x{j}").fill_nan(None) for j in range(20))
    )
    wide = [f"x{j}" for j in range(20)]
    common = dict(clock="t", gap_cap=6.0, weight="w")
    specs = [
        po.spec.ew_cov("cov", features=wide, half_life=50.0, **common),
        po.spec.ewridge(
            "grid",
            targets=[f"y{j}" for j in range(4)],
            features=wide[:5],
            half_life=[25.0, 100.0],
            ridge=[1e-6, 1e-2],
            emit_sigma=True,
            emit_metrics=True,
            conformal=0.9,
            resid_quantiles=[0.1, 0.9],
            **common,
        ),
        po.spec.ewridge(
            "grouped", targets=["y0"], features=wide[:3], group="g", half_life=30.0, **common
        ),
    ]
    return df, specs


def _series_bits(s: pl.Series) -> object:
    """A series as plain values that compare equal only when every value is
    the same to the bit: a float as its null mask and the bits of its values
    (``-0.0`` is not ``0.0``, and one NaN is not another), a list, an array
    or a struct as its null mask, its shape and its members' bits, anything
    else as its values."""
    nulls = s.is_null().to_list()
    if s.dtype in (pl.Float64, pl.Float32):
        view = np.uint64 if s.dtype == pl.Float64 else np.uint32
        return nulls, s.to_numpy().view(view).tolist()
    if isinstance(s.dtype, pl.List):
        return nulls, s.list.len().to_list(), _series_bits(s.explode())
    if isinstance(s.dtype, pl.Array):
        return nulls, _series_bits(s.explode())
    if isinstance(s.dtype, pl.Struct):
        return nulls, [_series_bits(f) for f in s.struct.unnest().iter_columns()]
    return s.to_list()


def _frame_bits(df: pl.DataFrame, prefix: str = "") -> dict[str, object]:
    """Every column of a frame, as :func:`_series_bits` gives it."""
    return {f"{prefix}{s.name}": _series_bits(s) for s in df.iter_columns()}


def _bits(out: pl.DataFrame) -> dict[str, object]:
    """Every field of every spec, as :func:`_series_bits` gives it."""
    got: dict[str, object] = {}
    for spec in out.columns:
        got |= _frame_bits(out[spec].struct.unnest(), f"{spec}.")
    return got


@pytest.mark.parametrize("chunks", [1, 2])
def test_a_spec_one_stream_serves_is_the_same_at_every_count(chunks):
    """Its copy of a run and the learning of the next run overlap on a pool
    of more than one thread and run in turn on one: every field is the same
    to the bit, null slots included, and so is the saved state."""
    df, specs = single_stream()
    size = -(-df.height // chunks)
    want = None
    for threads in (1, 2, 4, 8, None):
        bank = po.ModelBank(specs, threads=threads)
        out = pl.concat([bank.fit_predict(c) for c in df.iter_slices(size)]).drop(df.columns)
        got = (_bits(out), bank.save_bytes())
        if want is None:
            want = got
            # Compared over fields that hold numbers, not only nulls.
            for spec, prefix in (("cov", "corr_"), ("grid", "pred_y3")):
                field = next(f for f in out[spec].struct.fields if f.startswith(prefix))
                assert out[spec].struct.field(field).drop_nulls().len() > N_SINGLE // 2, field
            continue
        fields, state = got
        assert fields.keys() == want[0].keys(), threads
        for key, value in fields.items():
            assert value == want[0][key], (threads, chunks, key)
        assert state == want[1], (threads, chunks)


def test_the_count_is_not_state():
    bank = po.ModelBank(specs()[2:3], threads=8)
    bank.fit_predict(stream().head(50))
    assert bank.threads == 8
    data = bank.save_bytes()
    assert po.ModelBank.load(data).threads is None
    assert po.ModelBank.load(data, threads=2).threads == 2
    assert po.ModelBank.load_bytes(data, threads=3).threads == 3
    assert po.ModelBank.load(data, threads=2).save_bytes() == data
    # Pickle and deepcopy carry it beside the state.
    assert pickle.loads(pickle.dumps(bank)).threads == 8


def test_a_plan_and_a_frame_take_it(reference):
    want, _ = reference
    df = stream()
    lazy = po.fit_predict(df.lazy(), specs(), threads=2, chunk_size=10**6).collect()
    eager = df.online.fit_predict(specs(), threads=1)
    assert lazy.drop(df.columns).equals(want, null_equal=True)
    assert eager.drop(df.columns).equals(want, null_equal=True)
    bank = po.ModelBank(specs())
    bank.fit_predict(df.head(300))
    plain = po.predict(df.tail(300), bank)
    for threads in (1, 3):
        scored = po.predict(df.tail(300), bank, threads=threads)
        assert scored.equals(plain, null_equal=True)
        lazy = df.tail(300).lazy().online.predict(bank, threads=threads).collect()
        assert lazy.equals(plain, null_equal=True)
    # Scored as a copy: the caller's bank keeps its pool.
    assert bank.threads is None
    # A bank given as load_state keeps its count unless one is given.
    eight = po.ModelBank(specs(), threads=8)
    assert df.online.fit_predict(load_state=eight, threads=2).height == df.height


def test_the_gram_fits_take_it():
    rng = np.random.default_rng(2)
    df = pl.DataFrame({f"x{j}": rng.standard_normal(400) for j in range(6)}).with_columns(
        y=pl.Series(rng.standard_normal(400)), g=pl.Series([0, 1, 2, 3] * 100)
    )
    bank = po.ModelBank(
        [
            po.spec.ewridge(
                "r", targets=["y"], features=[f"x{j}" for j in range(6)], half_life=100.0, group="g"
            )
        ]
    )
    bank.fit_predict(df)
    grams = bank.gram("r")
    paths = po.gram.lars_paths(grams, max_active=3)
    fits = po.gram.solve_subsets(grams, [[0, 1], [2, 3]], ridge=1e-6)
    for threads in (1, 2, 8):
        again = po.gram.lars_paths(grams, max_active=3, threads=threads)
        for a, b in zip(paths, again, strict=True):
            for p, q in zip(a, b, strict=True):
                assert np.array_equal(p["coef"], q["coef"]) and p["active"] == q["active"]
        one = po.gram.lars_path(grams[0], max_active=3, threads=threads)
        assert np.array_equal(one["coef"], paths[0][0]["coef"])
        same = po.gram.solve_subsets(grams, [[0, 1], [2, 3]], ridge=1e-6, threads=threads)
        for a, b in zip(fits, same, strict=True):
            for p, q in zip(a, b, strict=True):
                assert np.array_equal(p["coef"], q["coef"])


BAD = [(0, ValueError), (-1, ValueError), (2.0, TypeError), (True, TypeError), ("2", TypeError)]


@pytest.mark.parametrize(("value", "error"), BAD, ids=[repr(v) for v, _ in BAD])
def test_a_count_that_is_not_one_is_refused_by_name_everywhere(tmp_path, value, error):
    spec = specs()[2]
    df = stream().head(10)
    bank = po.ModelBank([spec])
    data = bank.save_bytes()
    path = str(tmp_path / "bank.state")
    bank.save(path)
    message = _threads_refusal(repr(value))
    calls = {
        "ModelBank": lambda: po.ModelBank([spec], threads=value),
        "ModelBank.load": lambda: po.ModelBank.load(data, threads=value),
        "ModelBank.load_bytes": lambda: po.ModelBank.load_bytes(data, threads=value),
        "fit_predict(DataFrame)": lambda: po.fit_predict(df, [spec], threads=value),
        "fit_predict(LazyFrame)": lambda: po.fit_predict(df.lazy(), [spec], threads=value),
        "predict(DataFrame)": lambda: po.predict(df, bank, threads=value),
        "predict(LazyFrame)": lambda: po.predict(df.lazy(), bank, threads=value),
        "df.online.fit_predict": lambda: df.online.fit_predict([spec], threads=value),
        "df.online.predict": lambda: df.online.predict(path, threads=value),
        "lf.online.fit_predict": lambda: df.lazy().online.fit_predict([spec], threads=value),
        "lf.online.predict": lambda: df.lazy().online.predict(path, threads=value),
        "thread_pool_size": lambda: po.thread_pool_size(threads=value),
        "gram.lars_paths": lambda: po.gram.lars_paths([], threads=value),
        "gram.lars_path": lambda: po.gram.lars_path({}, threads=value),
        "gram.solve_subsets": lambda: po.gram.solve_subsets([], [[0]], threads=value),
        # The native side, reached without the Python check, says the same.
        "native ModelBank": lambda: native.ModelBank(_json([spec]), value),
        "native load_bytes": lambda: native.ModelBank.load_bytes(data, None, value),
        "native thread_pool_size": lambda: native.thread_pool_size(value),
    }
    for name, call in calls.items():
        with pytest.raises(error) as caught:
            call()
        assert str(caught.value) == message, name


@pytest.mark.parametrize("value", [np.int64(2), np.int32(2), np.uint8(2)], ids=repr)
def test_a_numpy_integer_is_a_count_everywhere(tmp_path, value):
    """Anything Python reads as an integer (`__index__`) is a count, numpy's
    included, on every surface and on the native side; numpy's bool is not
    one (task 269, D-6)."""
    spec = specs()[2]
    df = stream().head(10)
    bank = po.ModelBank([spec])
    data = bank.save_bytes()
    path = str(tmp_path / "bank.state")
    bank.save(path)
    grams = _grams()
    calls = {
        "ModelBank": lambda: po.ModelBank([spec], threads=value).threads,
        "ModelBank.load_bytes": lambda: po.ModelBank.load_bytes(data, threads=value).threads,
        "fit_predict(DataFrame)": lambda: po.fit_predict(df, [spec], threads=value),
        "fit_predict(LazyFrame)": lambda: po.fit_predict(
            df.lazy(), [spec], threads=value
        ).collect(),
        "df.online.predict": lambda: df.online.predict(path, threads=value),
        "lf.online.predict": lambda: df.lazy().online.predict(path, threads=value).collect(),
        "thread_pool_size": lambda: po.thread_pool_size(threads=value),
        "gram.lars_paths": lambda: po.gram.lars_paths(grams, threads=value),
        "gram.solve_subsets": lambda: po.gram.solve_subsets(grams, [[0, 1]], threads=value),
        "native ModelBank": lambda: native.ModelBank(_json([spec]), value).threads,
        "native thread_pool_size": lambda: native.thread_pool_size(value),
    }
    got = {name: call() for name, call in calls.items()}
    for name in ("ModelBank", "ModelBank.load_bytes", "thread_pool_size", "native ModelBank"):
        assert got[name] == 2 and type(got[name]) is int, name
    assert got["native thread_pool_size"] == 2
    message = _threads_refusal(repr(np.True_))
    for call in (
        lambda: po.ModelBank([spec], threads=np.True_),
        lambda: native.thread_pool_size(np.True_),
    ):
        with pytest.raises(TypeError) as caught:
            call()
        assert str(caught.value) == message


def _grams() -> list:
    rng = np.random.default_rng(3)
    df = pl.DataFrame({f"x{j}": rng.standard_normal(80) for j in range(3)}).with_columns(
        y=pl.Series(rng.standard_normal(80)), g=pl.Series([0, 1] * 40)
    )
    bank = po.ModelBank(
        [
            po.spec.ewridge(
                "r", targets=["y"], features=["x0", "x1", "x2"], half_life=50.0, group="g"
            )
        ]
    )
    bank.fit_predict(df)
    return bank.gram("r")


def test_none_is_the_default_pool_and_a_count_is_its_own():
    assert po.thread_pool_size(threads=None) == po.thread_pool_size()
    assert po.thread_pool_size(threads=3) == 3
    assert po.ModelBank(specs()[2:3]).threads is None


def _python(code: str, **env: str) -> str:
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=child.env(**env),
        cwd=str(REPO),
        check=False,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    return res.stdout.strip()


SMALL = (
    "import polars as pl, polars_online as po\n"
    "spec = po.spec.ewridge('m', targets=['y'], features=['x'], half_life=10.0)\n"
    "df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [2.0, 4.0, 6.0]})\n"
)


def test_the_variable_sizes_the_default_pool_and_nothing_given_a_count():
    out = _python(
        SMALL + "print(po.thread_pool_size(), po.thread_pool_size(threads=2))\n"
        "print(po.ModelBank([spec], threads=5).fit_predict(df).height)\n"
        "print(po.thread_pool_size())\n",
        POLARS_ONLINE_MAX_THREADS="3",
    )
    assert out.split() == ["3", "2", "3", "3"]
    # A variable that is not a count refuses the default pool, and a count
    # never reads it.
    out = _python(
        SMALL + "print(po.ModelBank([spec], threads=2).fit_predict(df).height)\n"
        "print(po.thread_pool_size(threads=4))\n"
        "try:\n"
        "    po.ModelBank([spec]).fit_predict(df)\n"
        "except ValueError as e:\n"
        "    print(str(e).split(' is ')[0])\n",
        POLARS_ONLINE_MAX_THREADS="eight",
    )
    assert out.split("\n") == ["3", "4", 'POLARS_ONLINE_MAX_THREADS="eight"']


# ---------------------------------------------------------------------------
# Every kind (task 278, batch 31)

#: Rows of the every-kind stream, fed in two chunks; the first is past the
#: 4,096 rows from which a spec's output columns are allocated and finished
#: in parallel.
N_KINDS = 6_000
FIRST_CHUNK = 5_000
#: Above every step of the clock but its few long gaps, which it caps.
GAP_CAP = 50.0
#: Each kind's case: its builder and the keywords beyond its ``MINIMAL``
#: entry. Every builder as ``MINIMAL`` builds it (``audit`` keeping its
#: pairs too), and ``sgd``'s Huber loss.
CASES: dict[str, tuple[str, dict[str, object]]] = {name: (name, {}) for name in MINIMAL} | {
    "audit": ("audit", {"pairs": True}),
    "sgd_huber": ("sgd", {"loss": "huber"}),
}
#: Every case, ``huber`` and ``sgd``'s Huber loss with the refit of their
#: first rows on (task 277).
SWEPT = sorted(CASES)
#: The essentials' case: a kind the sweeps above never ran, seeded inside
#: the stream.
ESSENTIAL_KIND = "kmeans"
#: The kinds that refuse a weight by name: a count, and a test whose every
#: learned row is one trial. ``rcov`` takes a weight of 0 or 1 only.
NO_WEIGHT = frozenset({"audit", "seqtest"})
WEIGHT = {"rcov": "w01"}
#: The kinds whose rows hold little beyond ``weight_sum``: what they learned
#: is read from the bank, and compared too.
READERS = {
    "marginal": lambda bank, spec: {"pairs": bank.marginal(spec)},
    "audit": lambda bank, spec: {
        table: bank.audit(spec, table=table) for table in ("columns", "pairs", "clock")
    },
    "rcov": lambda bank, spec: {"closed": bank.closed_groups(spec)},
}


def kind_frame(case: str) -> pl.DataFrame:
    """Four keys interleaved at random, an irregular clock with a few gaps
    past ``GAP_CAP``, weights with zeros among them (and as 0 or 1), a feature with nulls,
    two targets with a break in the first, and a key that only grows (rcov's
    block). ``ftrl`` reads 0/1 targets, ``ew_class`` a label."""
    rng = np.random.default_rng(31)
    i = np.arange(N_KINDS)
    step = np.where(rng.random(N_KINDS) < 0.005, 80.0, rng.exponential(1.0, N_KINDS))
    x0, x1 = rng.standard_normal((2, N_KINDS))
    y = 1.0 + 2.0 * x0 - x1 + 0.3 * rng.standard_normal(N_KINDS) + 3.0 * (i > N_KINDS // 2)
    z = -x0 + 0.5 * x1 + 0.5 * rng.standard_normal(N_KINDS)
    w = 0.5 + rng.random(N_KINDS)
    w[::37] = 0.0
    df = pl.DataFrame(
        {
            "t": np.cumsum(step),
            "x0": np.where(rng.random(N_KINDS) < 0.01, np.nan, x0),
            "x1": x1,
            "y": y,
            "z": z,
            "w": w,
            "w01": (w > 0).astype(float),
            "g": rng.choice(["a", "b", "c", "d"], N_KINDS),
            "blk": i // (N_KINDS // 4),
        }
    ).with_columns(pl.col("x0").fill_nan(None))
    builder = CASES[case][0]
    if builder == "ftrl":
        df = df.with_columns((pl.col("y") > 1).cast(pl.Float64), (pl.col("z") > 0).cast(pl.Float64))
    if builder == "ew_class":
        df = df.with_columns(y=pl.when(pl.col("y") > 1).then(pl.lit("a")).otherwise(pl.lit("b")))
    return df


def kind_specs(case: str) -> list[dict]:
    """The kind as ``MINIMAL`` builds it, and at its fullest (every output
    switch it takes, two targets where it takes any, a half-life grid where
    it decays), each on four groups with a clock and a weight, and ``coef``
    on every row where it has one."""
    builder, extra = CASES[case]
    stream: dict[str, object] = {
        "group": "blk" if builder == "rcov" else "g",
        "clock": "t",
        "gap_cap": GAP_CAP,
    }
    if builder not in NO_WEIGHT:
        stream["weight"] = WEIGHT.get(builder, "w")
    kw: dict[str, object] = {"targets": ["y"], "features": ["x0", "x1"], "half_life": 50.0}
    kw |= MINIMAL[builder] | extra | stream
    lone = getattr(po.spec, builder)("lone", **{k: v for k, v in kw.items() if v is not None})
    full = fullest(builder) | {"name": "full"} | stream
    full["model"] = full["model"] | extra
    if full.get("emit_drift"):
        # On a clock the drift detector's threshold is in clock time.
        full["drift_threshold"] = 20.0
    if any(f.startswith("coef") for f in po.spec.output_fields(lone)):
        lone["coef_every"] = full["coef_every"] = 0
    return [lone, full]


def run_kind(case: str, threads: int | None) -> tuple[pl.DataFrame, po.ModelBank]:
    df = kind_frame(case)
    bank = po.ModelBank(kind_specs(case), threads=threads)
    with warnings.catch_warnings():
        # The readiness notices of a short warm-up are expected.
        warnings.simplefilter("ignore")
        chunks = [df.slice(0, FIRST_CHUNK), df.slice(FIRST_CHUNK)]
        out = pl.concat([bank.fit_predict(c) for c in chunks]).drop(df.columns)
    return out, bank


def kind_bits(case: str, threads: int | None) -> tuple[po.ModelBank, dict[str, object], bytes]:
    """The bank, its output's every field and every table read from it as
    bits, and its saved state."""
    out, bank = run_kind(case, threads)
    state = bank.save_bytes()
    got = _bits(out) | {"rows": out.height}
    for spec, table, frame in _tables(case, bank):
        got |= _frame_bits(frame, f"{spec}.{table}.")
    return bank, got, state


def _tables(case: str, bank: po.ModelBank) -> list[tuple[str, str, pl.DataFrame]]:
    reader = READERS.get(CASES[case][0])
    if reader is None:
        return []
    return [(spec, t, f) for spec in ("lone", "full") for t, f in reader(bank, spec).items()]


def _learned(out: pl.DataFrame) -> dict[str, int]:
    """Per spec, the finite values of its float fields beyond ``weight_sum``:
    evidence that the comparison compares what the kind learned."""
    got = {}
    for spec in out.columns:
        fields = out[spec].struct.unnest()
        got[spec] = sum(
            int(s.is_finite().sum())
            for s in fields.iter_columns()
            if s.dtype == pl.Float64 and s.name != "weight_sum"
        )
    return got


def _kind(case: str):
    if case == ESSENTIAL_KIND:
        return case
    return pytest.param(
        case, marks=pytest.mark.extended(reason="a kind of the every-kind threads sweep")
    )


@pytest.mark.parametrize("case", [_kind(c) for c in SWEPT])
def test_every_kind_is_the_same_at_every_count(case):
    """Each kind, as ``MINIMAL`` builds it and at its fullest, on four
    groups: one thread, four and the process's pool give every field and
    every table read from the bank to the bit, null slots included, and the
    same saved state."""
    out, bank = run_kind(case, 1)
    want = _bits(out) | {"rows": out.height}
    state = bank.save_bytes()
    # Compared over what the kind learned: four groups in each spec, and
    # numbers in its fields, or in the tables its rows leave to the bank.
    tables = _tables(case, bank)
    for spec in ("lone", "full"):
        # rcov's groups are blocks, closed as the next begins.
        closed = sum(f.height for s, t, f in tables if (s, t) == (spec, "closed"))
        assert bank.groups(spec).height + closed == 4, (case, spec)
    learned = _learned(out)
    print(case, learned, {f"{s}.{t}": f.height for s, t, f in tables})
    if tables:
        assert all(f.height > 0 for _, _, f in tables), case
    else:
        assert min(learned.values()) > N_KINDS // 20, (case, learned)
    for spec, table, frame in tables:
        want |= _frame_bits(frame, f"{spec}.{table}.")
    for threads in (4, None):
        _, got, saved = kind_bits(case, threads)
        assert got.keys() == want.keys(), (case, threads)
        for key, value in got.items():
            assert value == want[key], (case, threads, key)
        assert saved == state, (case, threads)


def _cpu_by_thread() -> dict[int, float]:
    return {t.id: t.user_time + t.system_time for t in psutil.Process().threads()}


@pytest.mark.extended(reason="times the threads of a pool")
def test_a_pool_of_four_runs_the_work_on_more_than_one_thread():
    """The sweep above compares something: on a pool of four the kinds'
    (spec, group) tasks run on several threads, and on a pool of one on one.
    Read from the CPU time each of the process's threads accrues while the
    banks learn. Measured 2026-10-25 (macOS arm64, 14 cores, load 190):
    one thread ran 98-99% of the work at ``threads=1``; at ``threads=4``
    three ran 32-34% each and the fourth 2% (in a fresh process, over 20
    banks, all four ran 24-27%)."""
    df, specs = kind_frame("ewridge"), kind_specs("ewridge")
    chunks = [df.slice(0, FIRST_CHUNK), df.slice(FIRST_CHUNK)]
    shares = {}
    for threads in (1, 4):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            po.ModelBank(specs, threads=threads).fit_predict(df.head(50))
            before = _cpu_by_thread()
            for _ in range(5):
                bank = po.ModelBank(specs, threads=threads)
                for c in chunks:
                    bank.fit_predict(c)
            after = _cpu_by_thread()
        spent = [after[i] - before.get(i, 0.0) for i in after]
        total = sum(d for d in spent if d > 0)
        shares[threads] = sorted((d / total for d in spent if d > 0), reverse=True)
    print({k: [round(v, 3) for v in s[:6]] for k, s in shares.items()})
    assert shares[1][0] > 0.8, shares[1]
    assert sum(share > 0.1 for share in shares[4]) >= 2, shares[4]


def test_every_kind_is_swept():
    assert set(SWEPT) == set(CASES)
    assert {builder for builder, _ in CASES.values()} == set(MINIMAL)
