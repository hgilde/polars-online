"""`threads`: the pool a bank, a plan or a Gram fit runs on (docs/PLAN.md
task 253).

``threads=None`` is the process's pool, which ``POLARS_ONLINE_MAX_THREADS``
sizes when it is first used; an integer is a pool of that many threads,
shared by every holder of the count. The claims are bits: the output and the
saved state are the same at every count and every chunking, on a bank of
several specs and groups whose wide specs split their own work across the
pool (``gram_threads="auto"``, ``shards="auto"``). The count is not state: a
save carries none of it and a load takes its own. Every surface refuses a
count that is not an integer of 1 or more with the one message, and given a
count the bank never reads the variable. The Rust tests hold the pool
itself: the registry's sharing and release
(``crates/online-polars/src/pool.rs``) and that a bank of one thread runs
every task on it (``crates/online-polars/src/bank/threads_tests.rs``).
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import child
import polars_online as po
from polars_online import _polars_online as native
from polars_online._bank import _threads_refusal
from polars_online._spec import _json

TIER = "essential"

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
