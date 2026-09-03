"""How polars executes a Python IO source (`register_io_source`), measured.

The plan form, ``lf.online.fit_predict(specs)``, is such a source
(``python/polars_online/_frame.py``), and whether it can carry state out of
a query -- ``save_state=`` -- rests on facts about the engine that are not
documented: how often the source runs per query, whether two runs of it
overlap, what it sees when a later node fails, and what happens to a run
the caller abandons. ``docs/STATE-WORKFLOW.md`` records the answers; this
prints them for the installed polars. Package-independent: no bank, a toy
source of ``N`` rows in ``CH``-row chunks that logs its own life.

    uv run python scripts/io_source_semantics.py
"""

from __future__ import annotations

import gc
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from typing import Any

import polars as pl
from polars.io.plugins import register_io_source

N, CH = 1000, 100
SCHEMA = pl.Schema({"i": pl.Int64, "y": pl.Float64})


def toy(log: list[tuple[Any, ...]], slow: float = 0.0) -> pl.LazyFrame:
    """A source logging ``start`` (with the pushdowns it was handed), each
    ``yield``, its natural ``end``, ``n_rows`` reached, ``GeneratorExit`` and
    ``finally``, each stamped with the thread and the time."""

    def stamp(what: str, *rest: Any) -> None:
        log.append((what, threading.get_ident(), time.perf_counter(), *rest))

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        stamp(
            "start", with_columns, None if predicate is None else str(predicate), n_rows, batch_size
        )
        seen = 0
        try:
            for k in range(0, N, CH):
                if slow:
                    time.sleep(slow)
                df = pl.DataFrame(
                    {"i": range(k, k + CH), "y": [float(v) for v in range(k, k + CH)]}
                )
                if n_rows is not None:
                    take = min(df.height, n_rows - seen)
                    df, seen = df.head(take), seen + take
                if predicate is not None:
                    df = df.filter(predicate)
                if with_columns is not None:
                    df = df.select(with_columns)
                stamp("yield", k // CH)
                yield df
                if n_rows is not None and seen >= n_rows:
                    stamp("n_rows reached")
                    return
            stamp("end")
        except GeneratorExit:
            stamp("GeneratorExit")
            raise
        finally:
            stamp("finally")

    return register_io_source(source, schema=SCHEMA, validate_schema=True)


def count(log: list[tuple[Any, ...]], what: str) -> int:
    return sum(1 for e in log if e[0] == what)


def overlap(log: list[tuple[Any, ...]]) -> bool:
    """Two runs overlap when the second starts before the first ends."""
    starts = sorted(e[2] for e in log if e[0] == "start")
    ends = sorted(e[2] for e in log if e[0] in ("end", "n_rows reached"))
    return len(starts) > 1 and len(ends) > 0 and starts[1] < ends[0]


def row(name: str, log: list[tuple[Any, ...]], extra: str = "") -> None:
    print(
        f"{name:48s} runs={count(log, 'start')} ended={count(log, 'end')} "
        f"n_rows={count(log, 'n_rows reached')} closed={count(log, 'finally')} "
        f"threads={len({e[1] for e in log if e[0] == 'start'})} {extra}"
    )


def main() -> None:
    print(f"polars {pl.__version__}")
    d = tempfile.mkdtemp()
    pq = lambda name: os.path.join(d, name)  # noqa: E731

    print("\n# F1  one run per execution, and a plan used twice in a query runs twice (no CSE)")
    log: list[tuple[Any, ...]] = []
    lf = toy(log)
    lf.collect()
    lf.collect()
    row("collect() twice", log)
    log = []
    lf = toy(log)
    lf.join(lf, on="i").collect()
    row("lf.join(lf) .collect()", log)
    log = []
    lf = toy(log)
    lf.join(lf, on="i").collect(optimizations=pl.QueryOptFlags(comm_subplan_elim=True))
    row("lf.join(lf) .collect(cse on)", log)
    log = []
    lf = toy(log)
    pl.concat([lf, lf]).collect()
    row("pl.concat([lf, lf]).collect()", log)
    log = []
    lf = toy(log)
    pl.collect_all(
        [
            lf.sink_parquet(pq("a.parquet"), lazy=True),
            lf.select("y").sink_parquet(pq("b.parquet"), lazy=True),
        ]
    )
    row("collect_all([sink, sink]) one plan", log)

    print("\n# F2  those runs are concurrent (a slow source: 10 chunks x 20 ms)")
    for name, go in [
        ("self-join, in-memory engine", lambda lf: lf.join(lf, on="i").collect()),
        ("self-join, streaming engine", lambda lf: lf.join(lf, on="i").collect(engine="streaming")),
        (
            "collect_all two lazy sinks",
            lambda lf: pl.collect_all(
                [
                    lf.sink_parquet(pq("c.parquet"), lazy=True),
                    lf.select("y").sink_parquet(pq("d.parquet"), lazy=True),
                ]
            ),
        ),
    ]:
        log = []
        go(toy(log, slow=0.02))
        row(name, log, f"overlap={overlap(log)}")

    print("\n# F3  building and inspecting a plan runs nothing")
    log = []
    lf = toy(log)
    lf.collect_schema()
    lf.explain()
    lf.explain(engine="streaming")
    row("collect_schema / explain", log)

    print(
        "\n# F4  the pushdowns: head(n) is handed as n_rows and ends the run early; "
        "a filter is handed as the predicate"
    )
    log = []
    lf = toy(log)
    out = lf.head(5).collect()
    row("head(5)", log, f"rows={out.height} n_rows={[e[5] for e in log if e[0] == 'start']}")
    log = []
    lf = toy(log)
    out = lf.head(5).collect(engine="streaming")
    row("head(5), streaming engine", log, f"rows={out.height}")
    log = []
    lf = toy(log)
    out = lf.sort("y", descending=True).head(5).collect()
    row(
        "sort().head(5): slice not pushed",
        log,
        f"rows={out.height} n_rows={[e[5] for e in log if e[0] == 'start']}",
    )
    log = []
    lf = toy(log)
    out = lf.filter(pl.col("y") > 990).collect()
    row(
        "filter(y > 990)",
        log,
        f"rows={out.height} predicate={[e[4] for e in log if e[0] == 'start']}",
    )
    log = []
    lf = toy(log)
    lf.collect()
    lf.collect(engine="streaming")
    lf.sink_parquet(pq("e.parquet"))
    print(f"{'batch_size handed to the source':48s} {[e[6] for e in log if e[0] == 'start']}")

    print(
        "\n# F5  a node after the source failing: the source is still drained to its end "
        "first (slow source: 10 x 50 ms)"
    )
    for name, go in [
        ("collect()", lambda lf: lf.with_columns(pl.col("y").cast(pl.Int8, strict=True)).collect()),
        (
            "collect(engine='streaming')",
            lambda lf: lf.with_columns(pl.col("y").cast(pl.Int8, strict=True)).collect(
                engine="streaming"
            ),
        ),
        (
            "sink_parquet()",
            lambda lf: lf.with_columns(pl.col("y").cast(pl.Int8, strict=True)).sink_parquet(
                pq("f.parquet")
            ),
        ),
    ]:
        log = []
        t0 = time.perf_counter()
        try:
            go(toy(log, slow=0.05))
            raise AssertionError("expected the cast to fail")
        except pl.exceptions.InvalidOperationError:
            pass
        row(f"downstream error under {name}", log, f"raised after {time.perf_counter() - t0:.2f}s")
    print(
        f"{'the failed sink_parquet left a file':48s} {os.path.exists(pq('f.parquet'))}, "
        f"{os.path.getsize(pq('f.parquet'))} bytes"
    )

    print("\n# F6  an error inside the source")

    def bad(
        with_columns: Any, predicate: Any, n_rows: Any, batch_size: Any
    ) -> Iterator[pl.DataFrame]:
        yield pl.DataFrame({"i": [1], "y": [1.0]})
        raise RuntimeError("boom")

    try:
        register_io_source(bad, schema=SCHEMA).collect()
    except pl.exceptions.ComputeError as e:
        print(
            f"{'RuntimeError in the source surfaces as':48s} "
            f"ComputeError: {str(e).splitlines()[0][:70]}"
        )

    print(
        "\n# F7  an abandoned run: nothing is closed until the *plan* is dropped, "
        "and the source had run ahead"
    )
    log = []
    lf = toy(log)
    it = lf.collect_batches(chunk_size=CH)
    next(it)
    del it
    gc.collect()
    time.sleep(0.1)
    row("collect_batches: next(), del iterator", log, f"yielded={count(log, 'yield')}")
    del lf
    gc.collect()
    time.sleep(0.1)
    row("... then del plan", log, f"GeneratorExit={count(log, 'GeneratorExit')}")
    log = []
    lf = toy(log)
    for _ in lf.collect_batches(chunk_size=CH):
        break
    gc.collect()
    time.sleep(0.1)
    row("collect_batches: for .. break (plan alive)", log, f"yielded={count(log, 'yield')}")

    print("\n# F8  sink_batches (py-polars 1.44.1) drains the source whatever the callback returns")
    log = []
    lf = toy(log)
    got: list[int] = []

    def stop_after_first(df: pl.DataFrame) -> bool:
        got.append(df.height)
        return True

    lf.sink_batches(stop_after_first, chunk_size=CH, lazy=False)
    row("sink_batches(callback -> True)", log, f"callbacks={len(got)}")


if __name__ == "__main__":
    main()
