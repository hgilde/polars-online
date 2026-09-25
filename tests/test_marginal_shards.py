"""`marginal(shards=)`: a wide screen's pairs split across the bank's threads.

The pool runs groups and specs in parallel, so one wide `marginal` on one
group was one thread's work. `shards` cuts its pairs into ranges of features,
each stepped through a batch of held rows on a thread of its own
(docs/PLAN.md task 126). Every pair touches only its own cells, so the
numbers must be the unsplit model's to the bit, whatever the count, the
chunking, the save and load, and whatever the stream does to the model
between rows: zero weights, null targets, a skipped row, a session change,
a clock gap past `max_dclock`, groups, lags, bins and a window.
"""

from __future__ import annotations

import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po

P = 60


def frame(n: int = 1500, seed: int = 5) -> pl.DataFrame:
    """Two groups of a wide stream with every event the pairs must survive."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n, P)), axis=0) * 0.1 + rng.standard_normal((n, P))
    t = np.arange(n, dtype=float)
    t[n * 7 // 15 :] += 5e3  # a gap past max_dclock: the lags' ring empties
    w = rng.uniform(0.3, 2.0, n)
    w[::11] = 0.0
    cols = {f"x{j}": x[:, j] for j in range(P)}
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


FEATURES = [f"x{j}" for j in range(P)]

SHAPES = {
    "moments": {},
    "lags": {"lags": [1, 3, 7], "cross_lags": [1], "serial_rule": "geometric"},
    "bins": {"bins": 6, "bin_warm_rows": 50},
    "window": {"window": 40.0, "window_every": 4},
}


def spec(shape: str, shards, **kw):
    return po.spec.marginal(
        "m",
        targets=["y0", "y1", "y2"],
        features=FEATURES,
        clock="t",
        max_dclock=50.0,
        halflife=60.0,
        weight="w",
        group="g",
        session="s",
        session_gap=10.0,
        shards=shards,
        **SHAPES[shape],
        **kw,
    )


def run(shape: str, shards, df: pl.DataFrame, chunks: int = 1):
    bank = po.ModelBank([spec(shape, shards)])
    parts = [bank.fit_predict(c) for c in _split(df, chunks)]
    return pl.concat(parts), bank.marginal("m")


def _split(df: pl.DataFrame, chunks: int) -> list[pl.DataFrame]:
    edges = np.linspace(0, len(df), chunks + 1).astype(int)
    return [df[a:b] for a, b in zip(edges[:-1], edges[1:], strict=True)]


@pytest.mark.parametrize("shape", list(SHAPES))
def test_every_shard_count_gives_the_unsplit_numbers(shape):
    """The struct each row writes and every column of every pair, to the bit,
    for counts that divide the width, that do not, one per feature and more
    than that, and ``"auto"``; fed whole and in uneven chunks."""
    df = frame()
    out, pairs = run(shape, None, df)
    assert pairs.height == 2 * P * 3, "both groups, every pair"
    for shards in [2, 7, P, 500, "auto"]:
        for chunks in [1, 4]:
            got_out, got_pairs = run(shape, shards, df, chunks)
            assert got_out.equals(out, null_equal=True), (shape, shards, chunks)
            assert got_pairs.equals(pairs, null_equal=True), (shape, shards, chunks)


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


def test_the_cli_reads_shards(tmp_path, online_cli):
    """``shards = "auto"`` in the CLI's TOML: the unsplit output, to the bit."""
    df = frame(600)
    src, dst, cfg = tmp_path / "in.parquet", tmp_path / "out.parquet", tmp_path / "bank.toml"
    df.write_parquet(src)
    features = ", ".join(f'"{f}"' for f in FEATURES)
    cfg.write_text(
        "\n".join(
            [
                f'input = "{src.as_posix()}"',
                f'output = "{dst.as_posix()}"',
                "[[specs]]",
                'name = "m"',
                f"features = [{features}]",
                'targets = ["y0", "y1", "y2"]',
                'clock = "t"',
                "max_dclock = 50.0",
                "halflife = 60.0",
                'weight = "w"',
                'group = "g"',
                "[specs.model]",
                'type = "marginal"',
                'shards = "auto"',
                "lags = [1, 2]",
            ]
        )
    )
    subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
    want = po.ModelBank(
        [
            po.spec.marginal(
                "m",
                targets=["y0", "y1", "y2"],
                features=FEATURES,
                clock="t",
                max_dclock=50.0,
                halflife=60.0,
                weight="w",
                group="g",
                lags=[1, 2],
            )
        ]
    ).fit_predict(df)
    assert pl.read_parquet(dst)["m"].equals(want["m"], null_equal=True)
