"""Feed a bank from DuckDB: sort by the clock, stream, one cursor per plan.

DuckDB puts the rows in clock order and hands them over the Arrow PyCapsule
interface; polars scans that stream lazily; the bank's plan form fits it and
writes the result wherever a polars sink can. Nothing on the way is held
whole, and nothing needs pyarrow (docs/ARROW-SOURCES.md §3).

**The rule this example exists to show: one cursor per plan.**
``pl.scan_arrow_c_stream(rel)`` opens the query's result stream when the plan
is *built*, and a DuckDB connection holds one open result at a time. So a
second plan built on the same connection closes the first plan's stream, and
the first plan then yields no rows. ``con.cursor()`` gives each plan a
connection of its own. Breaking the rule here is at least loud: the empty plan
raises ``ConsumedSourceWarning``. Its ADBC twin is silent
(``examples/adbc_cursors.py``).

Measured on duckdb 1.5.5. DuckDB is a dev dependency of this project, not a
dependency of the package, so it is installed wherever the test suite runs:

    uv run python examples/duckdb_cursors.py

The example checks itself: it exits non-zero if a streamed fit stops equalling
an in-memory one, or if the trap stops reproducing.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import polars as pl

import polars_online as po

SPEC = po.spec.ewridge(
    "m",
    targets=["y"],
    features=["x0"],
    clock="t",
    halflife=600.0,
    max_dclock=30.0,
    min_periods=1.0,
)
N = 20_000
QUERY = "SELECT t, x0, y FROM ticks WHERE {where} ORDER BY t"
HALVES = {"first half": f"t < {N // 2}", "second half": f"t >= {N // 2}"}


def _check(ok: bool, message: str) -> None:
    """Stop with a non-zero exit if a claim this example makes stops holding."""
    if not ok:
        raise SystemExit(f"duckdb_cursors.py: {message}")


def _ticks(con: Any) -> None:
    """A clock and two columns, stored in shuffled order so that only the
    ``ORDER BY`` puts them in clock order. Generated in SQL: no data files."""
    con.execute(
        "CREATE TABLE ticks AS SELECT i::DOUBLE AS t, "
        "((i * 7919) % 1000) / 1000.0 AS x0, ((i * 104729) % 1000) / 500.0 AS y "
        f"FROM range({N}) AS r(i) ORDER BY random()"
    )


def _in_memory(con: Any, where: str) -> pl.DataFrame:
    """The reference: the same rows, fitted from a frame held whole."""
    rows = pl.DataFrame(con.sql(QUERY.format(where=where)))
    return po.ModelBank([SPEC]).fit_predict(rows)


def _same(got: pl.DataFrame, want: pl.DataFrame) -> bool:
    """Every float field of the output equal, nulls included."""
    g, w = got["m"].struct.unnest(), want["m"].struct.unnest()
    return got.height == want.height and all(
        g[c].equals(w[c], null_equal=True) for c in w.columns if w[c].dtype == pl.Float64
    )


def right_way(con: Any) -> None:
    """Two plans built up front and run later, each on a cursor of its own."""
    want = {name: _in_memory(con, where) for name, where in HALVES.items()}

    cursors = [con.cursor() for _ in HALVES]  # one cursor per plan
    plans = {
        name: pl.scan_arrow_c_stream(cur.sql(QUERY.format(where=where))).online.fit_predict([SPEC])
        for cur, (name, where) in zip(cursors, HALVES.items(), strict=True)
    }

    with tempfile.TemporaryDirectory() as d:
        # One plan streams straight to a file...
        path = Path(d) / "first_half.parquet"
        plans["first half"].sink_parquet(path)
        got = {"first half": pl.read_parquet(path)}
    # ...and the other is taken a batch at a time.
    got["second half"] = pl.concat(list(plans["second half"].collect_batches()))

    for cur in cursors:
        cur.close()
    for name in HALVES:
        _check(
            _same(got[name], want[name]), f"{name}: the streamed fit differs from an in-memory fit"
        )
        print(f"right way, {name}: {got[name].height} rows, equal to an in-memory fit")


def the_trap(con: Any) -> None:
    """Without the cursors: two plans on one connection, and the first empties."""
    query = QUERY.format(where="t >= 0")
    first = pl.scan_arrow_c_stream(con.sql(query)).online.fit_predict([SPEC])
    # Building this second plan opens its stream, and that closes the first's.
    second = pl.scan_arrow_c_stream(con.sql(query)).online.fit_predict([SPEC])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n_first = first.collect().height
        n_second = second.collect().height
    warned = any(issubclass(w.category, po.ConsumedSourceWarning) for w in caught)
    _check(
        n_first == 0 and n_second == N and warned,
        "two plans on one connection no longer empty the first; "
        "update docs/ARROW-SOURCES.md §3 and this example",
    )
    print(
        f"the trap, two plans on one connection: the first yielded {n_first} rows, "
        f"the second {n_second}, and ConsumedSourceWarning was raised"
    )


def main() -> int:
    try:
        import duckdb  # noqa: PLC0415
    except ImportError:
        print(
            "duckdb is not installed, so this example cannot run. It is a dev\n"
            "dependency of this project:\n"
            "    uv run python examples/duckdb_cursors.py"
        )
        return 0
    con = duckdb.connect()
    try:
        _ticks(con)
        right_way(con)
        the_trap(con)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
