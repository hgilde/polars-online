"""`marginal(shards=)`: a wide screen's pairs split across the bank's threads.

The pool runs groups and specs in parallel, so one wide `marginal` on one
group was one thread's work. `shards` cuts its pairs into ranges of features,
each stepped through a batch of held rows on a thread of its own
(docs/PLAN.md task 126). Every pair touches only its own cells, so the
numbers must be the unsplit model's to the bit, whatever the count, the
chunking, the save and load, and whatever the stream does to the model
between rows: zero weights, null targets, a skipped row, a session change,
a clock gap past `gap_cap`, groups, lags, bins and a window.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import polars as pl
import pytest

import child
import polars_online as po

TIER = "essential"

P = 60
#: A width at which ``"auto"`` splits in every shape on a pool of two or more
#: threads (`auto_shards_at_the_python_suites_widths` in marginal.rs pins the
#: counts): at ``P`` it splits the bins alone, and the other shapes' ``"auto"``
#: legs compared the unsplit model with itself (review 2026-09-26, F2).
P_AUTO = 1000


def features(p: int = P) -> list[str]:
    return [f"x{j}" for j in range(p)]


def frame(n: int = 1500, seed: int = 5, p: int = P) -> pl.DataFrame:
    """Two groups of a wide stream with every event the pairs must survive."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n, p)), axis=0) * 0.1 + rng.standard_normal((n, p))
    t = np.arange(n, dtype=float)
    t[n * 7 // 15 :] += 5e3  # a gap past gap_cap: the lags' ring empties
    w = rng.uniform(0.3, 2.0, n)
    w[::11] = 0.0
    cols = {f"x{j}": x[:, j] for j in range(p)}
    df = pl.DataFrame(
        {
            "t": t,
            "g": np.where(np.arange(n) % 3 == 0, "a", "b"),
            "s": ["s1"] * (n * 3 // 5) + ["s2"] * (n - n * 3 // 5),
            "w": w,
            "y0": x[:, 0] - 0.5 * x[:, 1] + rng.standard_normal(n),
            "y1": np.abs(x[:, 2]) + rng.standard_normal(n),
            "y2": np.where(x[:, 3] > 0, 1.0, -1.0) + 0.3 * rng.standard_normal(n),
            **cols,
        }
    )
    i = pl.int_range(pl.len())
    return df.with_columns(
        y1=pl.when(i % 5 == 2).then(None).otherwise(pl.col("y1")),
        y2=pl.when(i % 3 == 1).then(None).otherwise(pl.col("y2")),
        # A null feature: the row is skipped, as everywhere.
        x7=pl.when(i == 333).then(None).otherwise(pl.col("x7")),
    )


FEATURES = features()

SHAPES = {
    "moments": {},
    "lags": {"lags": [1, 3, 7], "cross_lags": [1], "serial_rule": "geometric"},
    "bins": {"bins": 6, "bin_warm_rows": 50},
    "window": {"window_size": 40.0, "max_rows_between_snapshots": 4},
    # Snapshots on the clock, four units apart (docs/PLAN.md task 162): the
    # flush before each must come where the clock says, in every group.
    "window_clock": {"window_size": 40.0, "window_every": 4.0},
}

#: The shapes for the ``"auto"`` legs: a window flushes before every snapshot,
#: so its row cap is what "auto" sizes by, and a snapshot every four rows is
#: never worth a split (review 2026-09-26, A1); every 128 rows is.
AUTO_SHAPES = {
    **{k: v for k, v in SHAPES.items() if k != "window_clock"},
    "window": {"window_size": 400.0, "max_rows_between_snapshots": 128},
}


def spec(shape: str, shards, p: int = P, shapes=SHAPES, **kw):
    settings = dict(
        targets=["y0", "y1", "y2"],
        features=features(p),
        clock="t",
        gap_cap=50.0,
        half_life=60.0,
        weight="w",
        group="g",
        session="s",
        session_gap=10.0,
        shards=shards,
        **shapes[shape],
    )
    settings.update(kw)
    return po.spec.marginal("m", **settings)


def run(shape: str, shards, df: pl.DataFrame, chunks: int = 1, **kw):
    bank = po.ModelBank([spec(shape, shards, **kw)])
    parts = [bank.fit_predict(c) for c in _split(df, chunks)]
    return pl.concat(parts), bank.marginal("m")


def _split(df: pl.DataFrame, chunks: int) -> list[pl.DataFrame]:
    edges = np.linspace(0, len(df), chunks + 1).astype(int)
    return [df[a:b] for a, b in zip(edges[:-1], edges[1:], strict=True)]


@pytest.mark.parametrize("shape", list(SHAPES))
def test_every_shard_count_gives_the_unsplit_numbers(shape):
    """The struct each row writes and every column of every pair, to the bit,
    for counts that divide the width, that do not, one per feature and more
    than that; fed whole and in uneven chunks."""
    df = frame()
    out, pairs = run(shape, None, df)
    assert pairs.height == 2 * P * 3, "both groups, every pair"
    for shards in [2, 7, P, 500]:
        for chunks in [1, 4]:
            got_out, got_pairs = run(shape, shards, df, chunks)
            assert got_out.equals(out, null_equal=True), (shape, shards, chunks)
            assert got_pairs.equals(pairs, null_equal=True), (shape, shards, chunks)


@pytest.mark.parametrize("shape", list(AUTO_SHAPES))
def test_auto_splits_at_a_width_that_keeps_the_pool_busy(shape):
    """``"auto"`` at a width every shape splits at, fed whole and in chunks:
    the unsplit numbers, to the bit (review 2026-09-26, F2)."""
    df = frame(p=P_AUTO)
    out, pairs = run(shape, None, df, p=P_AUTO, shapes=AUTO_SHAPES)
    assert pairs.height == 2 * P_AUTO * 3
    for chunks in [1, 3]:
        got_out, got_pairs = run(shape, "auto", df, chunks, p=P_AUTO, shapes=AUTO_SHAPES)
        assert got_out.equals(out, null_equal=True), (shape, chunks)
        assert got_pairs.equals(pairs, null_equal=True), (shape, chunks)


EXTRAS = {
    "embargo": ("lags", dict(embargo=7.0)),
    "grid": ("lags", dict(half_life=[40.0, 80.0])),
    # A window keeps no lags, so the reset under one is on the moments.
    "reset_window": (
        "moments",
        dict(session_gap=10.0, window_size=40.0, max_rows_between_snapshots=4),
    ),
}


@pytest.mark.parametrize("extra", list(EXTRAS))
def test_what_the_stream_does_between_rows_leaves_the_pairs_unsplit(extra):
    """A label delay (rows released later, in the stream's order), a half-life
    grid (one model per half-life, each holding its own rows) and a session
    reset under a window: the unsplit numbers, to the bit (review 2026-09-26,
    E missing 1, F missing 6)."""
    df = frame()
    shape, kw = EXTRAS[extra]
    out, pairs = run(shape, None, df, **kw)
    for chunks in [1, 3]:
        got_out, got_pairs = run(shape, 4, df, chunks, **kw)
        assert got_out.equals(out, null_equal=True), (extra, chunks)
        assert got_pairs.equals(pairs, null_equal=True), (extra, chunks)


def test_closed_groups_read_flushed_pairs_under_shards():
    """A group closed at a session change or by a monotone key reads its
    pairs between runs, after the flush: the unsplit bank's closed rows, to
    the bit, and the segment that follows a session boundary starts with
    nothing held (review 2026-09-26, E missing 2)."""
    # Groups in contiguous blocks, in order, each with its sessions inside.
    df = frame().sort("g", maintain_order=True)
    for close in ["session", "monotone"]:
        banks = {}
        for shards in [None, 4]:
            # The close is the session's prescription, so no gap beside it;
            # the monotone close reads no session at all.
            session = "s" if close == "session" else None
            bank = po.ModelBank(
                [spec("lags", shards, group_close=close, session=session, session_gap=None)]
            )
            for part in _split(df, 3):
                bank.fit_predict(part)
            banks[shards] = bank
        want, got = banks[None], banks[4]
        closed = want.closed_groups("m", drop=False)
        assert closed.height > 0, close
        assert "pair_target" in closed.columns
        assert got.closed_groups("m", drop=False).equals(closed, null_equal=True), close
        assert got.marginal("m").equals(want.marginal("m"), null_equal=True), close


def test_one_row_chunks_flush_every_row():
    """A chunk of one row is a run of one row: the end-of-run flush leaves
    nothing held, and the bins' warm-up replays across chunk ends (review
    2026-09-26, E missing 3)."""
    df = frame(300)
    out, pairs = run("bins", None, df)
    bank = po.ModelBank([spec("bins", 4)])
    got = pl.concat([bank.fit_predict(df[i : i + 1]) for i in range(len(df))])
    assert got.equals(out, null_equal=True)
    assert bank.marginal("m").equals(pairs, null_equal=True)


def test_scoring_never_holds_rows():
    """``predict`` steps no model, so it holds no rows: the pairs before and
    after a scored chunk are the same table, and the scores are the unsplit
    bank's (review 2026-09-26, E missing 5)."""
    df = frame()
    head, tail = df[:1000], df[1000:]
    want = po.ModelBank([spec("lags", None)])
    want.fit_predict(head)
    bank = po.ModelBank([spec("lags", 4)])
    bank.fit_predict(head)
    before = bank.marginal("m")
    scored = bank.predict(tail)
    assert bank.marginal("m").equals(before, null_equal=True)
    assert scored.equals(want.predict(tail), null_equal=True)


def test_a_save_inside_the_bin_warm_up_resumes_under_shards(tmp_path):
    """A state saved while the bins still hold their warm-up rows carries the
    hold (the batch is flushed before any save): resumed under a count, the
    unsplit bank's numbers (review 2026-09-26, E missing 6)."""
    df = frame()
    out, pairs = run("bins", None, df)
    bank = po.ModelBank([spec("bins", 5)])
    first = bank.fit_predict(df[:30])
    bank.save(tmp_path / "warm.state")
    resumed = po.ModelBank.load(tmp_path / "warm.state")
    second = resumed.fit_predict(df[30:])
    assert pl.concat([first, second]).equals(out, null_equal=True)
    assert resumed.marginal("m").equals(pairs, null_equal=True)


def test_a_large_snapshot_cadence_flushes_on_the_batch_alone():
    """A snapshot cadence longer than a batch: the flushes are the batch's,
    with a snapshot's flush now and then between (review 2026-09-26, E
    missing 7)."""
    df = frame()
    kw = dict(window_size=40.0, max_rows_between_snapshots=300)
    out, pairs = run("moments", None, df, **kw)
    got_out, got_pairs = run("moments", 4, df, 2, **kw)
    assert got_out.equals(out, null_equal=True)
    assert got_pairs.equals(pairs, null_equal=True)


def test_more_groups_than_threads_under_auto(tmp_path):
    """Forty groups on a pool of two threads, ``"auto"`` against none: a
    stream's flush forks inside the bank's own run over the groups, and a
    worker waiting on its shards may take up another group's run meanwhile;
    every number stays the unsplit one (review 2026-09-26, E missing 4)."""
    p = 300
    n = 1200
    df = frame(n, p=p).with_columns(
        g=pl.int_range(pl.len()).mod(40).cast(pl.String), s=pl.lit("s1")
    )
    data = tmp_path / "wide.parquet"
    df.write_parquet(data)
    script = tmp_path / "run.py"
    script.write_text(
        "import sys, hashlib\n"
        "import polars as pl\n"
        "import polars_online as po\n"
        f"features = [f'x{{j}}' for j in range({p})]\n"
        "df = pl.read_parquet(sys.argv[1])\n"
        "out = []\n"
        "for shards in [None, 'auto']:\n"
        "    spec = po.spec.marginal('m', targets=['y0', 'y1', 'y2'], features=features,\n"
        "        clock='t', gap_cap=50.0, half_life=60.0, weight='w', group='g',\n"
        "        lags=[1, 3, 7], cross_lags=[1], shards=shards)\n"
        "    bank = po.ModelBank([spec])\n"
        "    frames = [bank.fit_predict(df[:500]), bank.fit_predict(df[500:])]\n"
        "    h = hashlib.sha256()\n"
        "    for f in frames + [bank.marginal('m')]:\n"
        "        h.update(str(f.to_dict(as_series=False)).encode())\n"
        "    out.append(h.hexdigest())\n"
        "print(' '.join(out))\n"
    )
    res = subprocess.run(
        [sys.executable, str(script), str(data)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=child.env(POLARS_ONLINE_MAX_THREADS="2"),
        check=True,
    )
    unsplit, auto = res.stdout.split()
    assert unsplit == auto


def test_a_saved_bank_resumes_whatever_its_count(tmp_path):
    """``shards`` is a setting, not state: a bank saved under one count and
    resumed reads the same pairs as one that never split."""
    df = frame()
    head, tail = df[:800], df[800:]
    tables = []
    for i, shards in enumerate([None, 5, "auto"]):
        bank = po.ModelBank([spec("lags", shards)])
        bank.fit_predict(head)
        bank.save(tmp_path / f"{i}.state")
        resumed = po.ModelBank.load(tmp_path / f"{i}.state")
        resumed.fit_predict(tail)
        tables.append(resumed.marginal("m"))
    for t in tables[1:]:
        assert t.equals(tables[0], null_equal=True)


def test_a_saved_bank_resumes_under_another_count(tmp_path):
    """A bank saved under one count loads under the count of the specs given
    to ``load``, and keeps the saved one when given none: the setting is the
    caller's, not the file's (review 2026-09-26, F3: ``load`` compared the
    counts and refused the specs)."""
    df = frame()
    head, tail = df[:800], df[800:]
    want = po.ModelBank([spec("lags", None)])
    want.fit_predict(df)
    bank = po.ModelBank([spec("lags", 3)])
    bank.fit_predict(head)
    bank.save(tmp_path / "three.state")
    for shards in [None, "auto", 7]:
        resumed = po.ModelBank.load(tmp_path / "three.state", specs=[spec("lags", shards)])
        assert resumed.specs[0]["model"].get("shards") == shards
        resumed.fit_predict(tail)
        assert resumed.marginal("m").equals(want.marginal("m"), null_equal=True), shards
    kept = po.ModelBank.load(tmp_path / "three.state")
    assert kept.specs[0]["model"]["shards"] == 3


def test_a_lazy_frame_is_split_the_same_way():
    """``fit(lf)`` streams the frame in batches through the same bank."""
    df = frame(900)
    want = po.ModelBank([spec("bins", None)])
    want.fit(df.lazy())
    got = po.ModelBank([spec("bins", 4)])
    got.fit(df.lazy())
    assert got.marginal("m").equals(want.marginal("m"), null_equal=True)


@pytest.mark.parametrize(
    ("shards", "error", "message"),
    [
        (0, ValueError, "shards must be >= 1"),
        ("many", ValueError, 'shards must be a number of shards of at least 1 or "auto"'),
        (2.5, TypeError, "shards must be an int or a str"),
    ],
)
def test_a_count_that_means_nothing_is_refused(shards, error, message):
    with pytest.raises(error, match=message):
        spec("moments", shards)


@pytest.mark.parametrize("shards", ['"auto"', "4"])
def test_the_cli_reads_shards(tmp_path, online_cli, shards):
    """``shards`` in the CLI's TOML: the unsplit output, and -- since the
    output struct is ``weight_sum`` alone, which no shard computes -- the pairs of
    the state it saves, to the bit (review 2026-09-26, E1)."""
    df = frame(600)
    src, dst, cfg = tmp_path / "in.parquet", tmp_path / "out.parquet", tmp_path / "bank.toml"
    state = tmp_path / "bank.state"
    df.write_parquet(src)
    features_toml = ", ".join(f'"{f}"' for f in FEATURES)
    cfg.write_text(
        "\n".join(
            [
                f'input = "{src.as_posix()}"',
                f'output = "{dst.as_posix()}"',
                f'save_state = "{state.as_posix()}"',
                "[[specs]]",
                'name = "m"',
                f"features = [{features_toml}]",
                'targets = ["y0", "y1", "y2"]',
                'clock = "t"',
                "gap_cap = 50.0",
                "half_life = 60.0",
                'weight = "w"',
                'group = "g"',
                "[specs.model]",
                'type = "marginal"',
                f"shards = {shards}",
                "lags = [1, 2]",
            ]
        )
    )
    subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
    want_bank = po.ModelBank(
        [
            po.spec.marginal(
                "m",
                targets=["y0", "y1", "y2"],
                features=FEATURES,
                clock="t",
                gap_cap=50.0,
                half_life=60.0,
                weight="w",
                group="g",
                lags=[1, 2],
            )
        ]
    )
    want = want_bank.fit_predict(df)
    assert pl.read_parquet(dst)["m"].equals(want["m"], null_equal=True)
    saved = po.ModelBank.load(state)
    assert saved.marginal("m").equals(want_bank.marginal("m"), null_equal=True)
