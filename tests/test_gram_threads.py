"""`gram_threads` on `ewridge` and `ew_cov` (docs/PLAN.md tasks 225 and 253).

The option runs the same model on more threads: a block's merge under
`gram_block_rows` is cut into pieces fixed by the width alone, each summed by
one thread in one order, and the per-row rank-one update is split by rows. So
the claims are bits, not tolerances: every output field and every array of
`gram()` is the same at one thread and at several, and at ``"auto"``, the
default, which shares the pool out among the groups, on banks of one, two
and eight threads (`ModelBank(threads=)`), chunk invariance holds at each
count, a save carries the setting and resumes to the bit, a spec that leaves
it out writes nothing new into a state, and a count of zero, or a word other
than ``"auto"``, is refused.

Two widths, each past the point where its path leaves the calling thread
(`crates/online-core/src/ewcov/par.rs`): 300 columns for a block's merge,
which is then more than one piece, and 730 for the per-row update, which
splits from 725. The Rust tests there hold the pieces themselves, and that
the work does reach other threads.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

#: Wide enough that a block's merge is several pieces.
K = 300
#: Wide enough that a row's update is split across threads.
K_ROW = 730
HL = 200.0


def stream(k: int, n: int, seed: int = 3) -> pl.DataFrame:
    """Features on an offset, two targets null on different rows (so
    `own_rows` splits the Gram), a zero-weight first row and more inside, a
    gap of three half-lives two thirds of the way in, and a session change
    at three quarters."""
    rng = np.random.default_rng(seed)
    X = 50.0 + rng.standard_normal((n, k))
    beta = rng.standard_normal(k) / np.sqrt(k)
    y = X @ beta + 0.1 * rng.standard_normal(n)
    z = X @ beta[::-1] + 0.1 * rng.standard_normal(n)
    y[7::19] = np.nan
    z[11::23] = np.nan
    w = 0.5 + rng.random(n)
    w[0] = 0.0
    w[::29] = 0.0
    t = np.arange(float(n))
    t[2 * n // 3 :] += 3 * HL
    df = pl.DataFrame({f"x{j}": X[:, j] for j in range(k)})
    change = 3 * n // 4
    return df.with_columns(
        y=pl.Series(y).fill_nan(None),
        z=pl.Series(z).fill_nan(None),
        w=pl.Series(w),
        t=pl.Series(t),
        s=pl.Series(["a"] * change + ["b"] * (n - change)),
    )


def ridge(threads: int | str | None, block: int | None, k: int = K, **kw) -> dict:
    opts = dict(
        targets=["y", "z"],
        features=[f"x{j}" for j in range(k)],
        clock="t",
        gap_cap=3 * HL,
        half_life=HL,
        weight="w",
        ridge=1e-3,
        min_weight=50.0,
        solve_every=50.0,
        max_rows_between_solves=100,
        coef_every=0,
        gram_block_rows=block,
        gram_threads=threads,
    )
    opts.update(kw)
    return po.spec.ewridge("m", **opts)


def cov(threads: int | str | None, k: int = K, **kw) -> dict:
    opts = dict(
        features=[f"x{j}" for j in range(k)],
        clock="t",
        gap_cap=3 * HL,
        half_life=HL,
        weight="w",
        stats=["mean", "var"],
        gram_threads=threads,
    )
    opts.update(kw)
    return po.spec.ew_cov("m", **opts)


def run(
    spec: dict, df: pl.DataFrame, chunks: int = 1, threads: int | None = None
) -> tuple[pl.DataFrame, po.ModelBank]:
    bank = po.ModelBank([spec], threads=threads)
    size = -(-df.height // chunks)
    out = pl.concat([bank.fit_predict(c) for c in df.iter_slices(size)])
    return out.select("m").unnest("m"), bank


def gram_bits(bank: po.ModelBank) -> list:
    """Every float array and scalar of every Gram, as its bits."""
    out = []
    for g in bank.gram("m"):
        for key in sorted(g):
            v = g[key]
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                out.append((key, v.view(np.uint64).tolist()))
            elif isinstance(v, float):
                out.append((key, np.float64(v).view(np.uint64).item()))
            else:
                out.append((key, v))
    return out


#: Each case: the spec at a thread count, the width, the rows, and whether
#: `own_rows` splits its Gram.
SPECS = {
    "ewridge-blocked": (lambda t: ridge(t, 64), K, 600, True),
    "ewridge-pairwise": (lambda t: ridge(t, 64, target_gaps="pairwise"), K, 600, False),
    "ewridge-twin": (
        lambda t: ridge(
            t, 64, session="s", session_gap=1.0, session_shrink=0.5, long_half_life=4 * HL
        ),
        K,
        600,
        True,
    ),
    "ewridge-per-row": (lambda t: ridge(t, None, k=K_ROW), K_ROW, 240, True),
    # Narrow: the update never leaves the calling thread, at any setting.
    "ewridge-narrow": (lambda t: ridge(t, None, k=20), 20, 300, True),
    "ew_cov": (lambda t: cov(t, k=K_ROW), K_ROW, 240, False),
}


@pytest.mark.parametrize("kind", SPECS)
def test_every_thread_count_gives_the_same_bits(kind):
    """Each ``gram_threads`` -- a count, ``"auto"`` and none -- on a bank of
    one thread, two and eight, against one thread on the default pool."""
    make, k, n, splits = SPECS[kind]
    df = stream(k, n)
    one, bank_one = run(make(1), df)
    assert len(bank_one.gram("m")) >= (2 if splits else 1)
    for pool in (None, 1, 2, 8):
        for threads in (None, 1, 3, "auto", 8):
            many, bank_many = run(make(threads), df, threads=pool)
            case = f"gram_threads={threads!r}, threads={pool!r}"
            assert one.equals(many, null_equal=True), case
            assert gram_bits(bank_one) == gram_bits(bank_many), case
    if kind.startswith("ewridge"):
        # The fit ran: rows past the warm-up are predicted.
        assert one["pred_y"].drop_nulls().len() > n // 5


@pytest.mark.parametrize("kind", ["ewridge-blocked", "ewridge-per-row", "ew_cov"])
@pytest.mark.parametrize("threads", [1, 8, "auto"])
def test_chunk_invariance_at_each_thread_count(kind, threads):
    """One chunk, seven, and one row a chunk."""
    make, k, n, _ = SPECS[kind]
    df = stream(k, n)
    whole, bank = run(make(threads), df)
    for chunks in (7, n):
        cut, cut_bank = run(make(threads), df, chunks)
        assert whole.equals(cut, null_equal=True), f"{chunks} chunks"
        assert gram_bits(bank) == gram_bits(cut_bank), f"{chunks} chunks"


def test_a_save_carries_the_setting_and_resumes_to_the_bit(tmp_path):
    df = stream(K, 600)
    whole, _ = run(ridge(8, 64), df)
    a = po.ModelBank([ridge(8, 64)])
    a.fit_predict(df.slice(0, 301))
    a.save(tmp_path / "threads.state")
    b = po.ModelBank.load(tmp_path / "threads.state")
    assert '"gram_threads":8' in b.write_json().replace(" ", "")
    rest = b.fit_predict(df.slice(301)).select("m").unnest("m")
    assert rest.equals(whole.slice(301), null_equal=True)


@pytest.mark.parametrize("kind", ["ewridge", "ew_cov"])
def test_a_save_resumes_under_another_count_to_the_bit(tmp_path, kind):
    """The count is configuration, not state: a bank saved at 8 threads
    resumes at 2, or with the key left out, through ``load(specs=)`` and the
    lazy frame's ``load_state``, to the bit of the unbroken run, and runs
    the caller's count (review 6, D-1: the load compared it, as review
    2026-09-26 F3 found for marginal's ``shards``, and refused the specs).
    The command line's door is ``crates/online-cli/tests/run.rs``."""
    make = (lambda t: ridge(t, 64)) if kind == "ewridge" else (lambda t: cov(t, k=K))
    df = stream(K, 600)
    head, tail = df.slice(0, 301), df.slice(301)
    whole, want = run(make(None), df)
    a = po.ModelBank([make(8)])
    a.fit_predict(head)
    a.save(tmp_path / "threads.state")
    tail.write_parquet(tmp_path / "tail.parquet")
    for threads in (2, "auto", None):
        b = po.ModelBank.load(tmp_path / "threads.state", specs=[make(threads)])
        assert b.specs[0]["model"].get("gram_threads") == threads
        rest = b.fit_predict(tail).select("m").unnest("m")
        assert rest.equals(whole.slice(301), null_equal=True), threads
        assert gram_bits(b) == gram_bits(want), threads
        lazy = (
            pl.scan_parquet(tmp_path / "tail.parquet")
            .online.fit_predict([make(threads)], load_state=tmp_path / "threads.state")
            .collect()
        )
        assert lazy.select("m").unnest("m").equals(whole.slice(301), null_equal=True), threads
    kept = po.ModelBank.load(tmp_path / "threads.state")
    assert kept.specs[0]["model"]["gram_threads"] == 8


@pytest.mark.parametrize("value", [0, -1])
def test_the_builder_names_the_floor_rust_holds(value):
    """The builder's floor is the one Rust refuses below, 1 (review 6, D-3:
    it said ``>= 0``, and 0 was then refused by Rust as ``>= 1``)."""
    for make in (lambda: ridge(value, 64), lambda: cov(value)):
        with pytest.raises(ValueError, match="gram_threads must be >= 1"):
            make()


def test_a_spec_without_it_writes_nothing_new():
    """Left out, the key is not written: a state that does not use it has
    the bytes it had before the option existed (no schema change)."""
    for spec in (ridge(None, 64), cov(None)):
        assert "gram_threads" not in po.ModelBank([spec]).write_json()
    for spec in (ridge(2, 64), cov(2)):
        doc = json.loads(po.ModelBank([spec]).write_json())
        assert '"gram_threads": 2' in json.dumps(doc)


@pytest.mark.parametrize("make", [lambda: ridge(0, 64), lambda: cov(0)], ids=["ewridge", "ew_cov"])
def test_zero_threads_is_refused(make):
    with pytest.raises(ValueError, match="gram_threads must be >= 1"):
        po.ModelBank([make()])


@pytest.mark.parametrize("build", [ridge, cov], ids=["ewridge", "ew_cov"])
def test_auto_is_the_one_word_and_is_written_as_given(build):
    """``"auto"`` is taken and written as given; another word, or a bool, is
    refused by name, in the shape ``shards`` refuses one, by the builder and
    by the bank reading a hand-written spec."""
    make = (lambda t: build(t, 64)) if build is ridge else build
    spec = make("auto")
    assert spec["model"]["gram_threads"] == "auto"
    assert po.ModelBank([spec]).specs[0]["model"]["gram_threads"] == "auto"
    with pytest.raises(
        ValueError, match='gram_threads must be an int or one of "auto"; got "AUTO"'
    ):
        make("AUTO")
    with pytest.raises(TypeError, match='gram_threads must be an int or one of "auto", got bool'):
        make(True)
    for bad in ("x", True, -1):
        hand = {**spec, "model": {**spec["model"], "gram_threads": bad}}
        with pytest.raises(ValueError, match="gram_threads"):
            po.ModelBank([hand])


#: Groups a bank runs side by side, for ``"auto"``'s share of the pool: one
#: (the whole pool), four, and 28 and 64 (one thread each on any pool up to
#: 28 threads).
GROUPS = (1, 4, 28, 64)

#: Each case: the spec at a ``gram_threads`` and grouped by ``g``, the
#: width, the rows.
GROUPED = {
    "ewridge-blocked": (lambda t: ridge(t, 64, group="g"), K, 600),
    "ew_cov": (lambda t: cov(t, k=K_ROW, group="g"), K_ROW, 240),
}


@pytest.mark.parametrize("kind", GROUPED)
@pytest.mark.parametrize("groups", GROUPS)
def test_auto_shares_the_pool_and_moves_no_bit(kind, groups):
    """``"auto"`` splits a group's update only into the threads the bank's
    other tasks leave idle, ``max(1, threads // tasks)``. Whichever share
    that gives, the output and every Gram are the bits of one thread, on
    pools of two and eight threads and the default, over 1, 7 and 600
    chunks. Which share is taken is held in Rust, from the rule
    (``crates/online-polars/src/bank/threads_tests.rs``)."""
    make, k, n = GROUPED[kind]
    df = stream(k, n).with_columns(g=pl.int_range(pl.len()) % groups)
    one, bank_one = run(make(1), df)
    assert bank_one.groups("m").height == groups
    for pool in (2, 8, None):
        for chunks in (1, 7, 600):
            many, bank_many = run(make("auto"), df, chunks, threads=pool)
            case = f"threads={pool!r}, {chunks} chunks"
            assert one.equals(many, null_equal=True), case
            assert gram_bits(bank_one) == gram_bits(bank_many), case
