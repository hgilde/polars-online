"""Feed a bank from any ADBC database: one cursor per plan, or wrong answers.

ADBC is one driver manager in front of Postgres, Snowflake, BigQuery and
Flight SQL (docs/ARROW-SOURCES.md §2). A cursor's ``fetch_arrow()`` returns a
stream carrying the Arrow PyCapsule interface, so polars scans it lazily, the
bank's plan form fits it, and nothing needs pyarrow.

**The rule this example exists to show: one cursor per plan** -- and here,
unlike DuckDB, breaking it is silent. Re-execute a cursor while a plan built
on its earlier stream is still uncollected, and the plans come back wrong:
rows missing from the newer one, with no error and no warning.
``ConsumedSourceWarning`` cannot see it, because the plan still yields rows.
Measured on the SQLite driver, adbc-driver-manager 1.12.0; another driver may
refuse the second execute instead.

Two spellings to avoid, both measured without pyarrow:
``pl.read_database(..., iter_batches=True)`` needs pyarrow for ADBC, and
``cursor.fetch_polars()`` goes through ``pl.from_arrow``, which polars 2.0
changes to return a Series. ``pl.scan_arrow_c_stream(cursor.fetch_arrow())``
streams, and needs neither.

ADBC is a dev dependency of this project, not a dependency of the package:

    uv run python examples/adbc_cursors.py

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
        raise SystemExit(f"adbc_cursors.py: {message}")


def _ticks() -> pl.DataFrame:
    """A clock and two columns, shuffled so that only the ``ORDER BY`` puts
    them back in clock order. Generated here: no data files."""
    i = pl.int_range(N, eager=True)
    frame = pl.DataFrame(
        {
            "t": i.cast(pl.Float64),
            "x0": ((i * 7919) % 1000) / 1000.0,
            "y": ((i * 104729) % 1000) / 500.0,
        }
    )
    return frame.sample(fraction=1.0, shuffle=True, seed=0)


def _in_memory(ticks: pl.DataFrame, where: str) -> pl.DataFrame:
    """The reference: the same rows, sorted, fitted from a frame held whole."""
    rows = ticks.filter(pl.sql_expr(where)).sort("t")
    return po.ModelBank([SPEC]).fit_predict(rows)


def _same(got: pl.DataFrame, want: pl.DataFrame) -> bool:
    """Every float field of the output equal, nulls included."""
    g, w = got["m"].struct.unnest(), want["m"].struct.unnest()
    return got.height == want.height and all(
        g[c].equals(w[c], null_equal=True) for c in w.columns if w[c].dtype == pl.Float64
    )


def right_way(con: Any, ticks: pl.DataFrame) -> None:
    """Two plans built up front and run later, each on a cursor of its own."""
    want = {name: _in_memory(ticks, where) for name, where in HALVES.items()}

    cursors = [con.cursor() for _ in HALVES]  # one cursor per plan
    plans = {}
    for cur, (name, where) in zip(cursors, HALVES.items(), strict=True):
        cur.execute(QUERY.format(where=where))
        plans[name] = pl.scan_arrow_c_stream(cur.fetch_arrow()).online.fit_predict([SPEC])

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


def the_trap(con: Any, ticks: pl.DataFrame) -> None:
    """Without separate cursors: one cursor, re-executed before its first plan
    is collected. The newer plan comes back short, yet sorted and plausible."""
    where = "t >= 0"
    want = _in_memory(ticks, where)
    cur = con.cursor()
    cur.execute(QUERY.format(where=where))
    older = pl.scan_arrow_c_stream(cur.fetch_arrow()).online.fit_predict([SPEC])  # not collected
    cur.execute(QUERY.format(where=where))  # the same cursor, re-executed
    newer = pl.scan_arrow_c_stream(cur.fetch_arrow()).online.fit_predict([SPEC])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = newer.collect()
    _check(
        not _same(got, want) and not caught,
        "re-executing a cursor no longer corrupts a plan silently; "
        "update docs/ARROW-SOURCES.md §2 and this example",
    )
    print(
        f"the trap, one cursor re-executed: the newer plan returned {got.height} of {N} rows, "
        f"clock in order: {got['t'].is_sorted()}, with no error and no warning"
    )
    try:
        stale = older.collect()
        print(
            f"  the older plan then returned {stale.height} rows, "
            f"clock in order: {stale['t'].is_sorted()}"
        )
    except Exception as err:  # reported, not relied on: this half varies by driver
        print(f"  the older plan then raised {type(err).__name__}")
    cur.close()


def main() -> int:
    try:
        from adbc_driver_sqlite import dbapi  # noqa: PLC0415
    except ImportError:
        print(
            "adbc-driver-sqlite is not installed, so this example cannot run. It is\n"
            "a dev dependency of this project:\n"
            "    uv run python examples/adbc_cursors.py"
        )
        return 0
    ticks = _ticks()
    with tempfile.TemporaryDirectory() as d:
        con = dbapi.connect(str(Path(d) / "ticks.db"))
        try:
            ticks.write_database("ticks", connection=con, engine="adbc")
            con.commit()
            right_way(con, ticks)
            the_trap(con, ticks)
        finally:
            con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
