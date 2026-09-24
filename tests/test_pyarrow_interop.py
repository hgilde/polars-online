"""pyarrow, the reference Arrow implementation, reading a bank and feeding one.

The README, ``ModelBank.fit_predict_arrow``'s docstring and docs/ARROW-SOURCES.md
say pyarrow reads a bank's Arrow output, and that a pyarrow reader streams into
a bank. Until 2026-09-24 nothing checked either, because pyarrow was not
installed. It is now, with the dev group, for these tests alone. The package
still depends on polars alone, and tests/conftest.py makes pyarrow unimportable
everywhere else in the suite, so each test here runs its pyarrow half in a
fresh interpreter, which that block does not reach. Each such run starts by
reporting the pyarrow it imported, so none of these can pass without it.

The inputs are tests/test_arrow_capsule.py's: two groups, nulls in a feature
and in the target, and the Enum and nested-list fields a spec emits.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

#: What every child imports. `same` compares every field that is not reported
#: on a chunk's cadence: `coef` and `support_coef` appear on each group's last
#: row in each chunk, by design, so a chunked run matches them only there.
PRELUDE = f"""
import json, sys, warnings
sys.path.insert(0, {str(TESTS)!r})
import polars as pl
import pyarrow as pa
import polars_online as po
from test_arrow_capsule import SPEC, frame

df = frame(200)
want = po.ModelBank([SPEC]).fit_predict(df)["m"].struct.unnest()
result = {{"pyarrow": pa.__version__}}

def fresh():
    return po.ModelBank([SPEC]).fit_predict_arrow(df)[0]

def same(got, cadence_too=True):
    cols = [c for c in want.columns if cadence_too or c not in ("coef", "support_coef")]
    return list(got.columns) == list(want.columns) and all(
        got[c].equals(want[c], null_equal=True) for c in cols
    )
"""


def _child(body: str) -> dict:
    """Run `body` after PRELUDE in a fresh interpreter; it fills `result`."""
    code = PRELUDE + textwrap.dedent(body) + "\nprint(json.dumps(result))\n"
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=TESTS.parent,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["pyarrow"], "the child ran without pyarrow"
    return out


def test_this_session_runs_without_pyarrow() -> None:
    """The block in tests/conftest.py is in force here, while the children
    of the tests below find pyarrow installed. Both halves are what make
    the rest of the suite a test of the package as its users install it."""
    with pytest.raises(ModuleNotFoundError):
        import pyarrow  # noqa: F401
    with pytest.raises(ModuleNotFoundError):
        importlib.util.find_spec("pyarrow")
    assert "pyarrow" not in sys.modules


def test_pyarrow_takes_the_output_through_each_entry_point() -> None:
    """Measured on pyarrow 25.0.1: each of these takes an ``ArrowStruct`` as
    it is, and the values are ``fit_predict``'s, the ``withheld_reason`` Enum
    and the nested ``coef`` list included. Read back into polars, the dtypes
    are the same too."""
    got = _child(
        """
        entry = {
            "pa.array": pa.array,
            "pa.chunked_array": pa.chunked_array,
            "pa.record_batch": pa.record_batch,
            "pa.table": pa.table,
        }
        for label, read in entry.items():
            obj = read(fresh())
            back = (
                pl.Series(obj).struct.unnest()
                if isinstance(obj, (pa.Array, pa.ChunkedArray))
                else pl.DataFrame(obj)
            )
            result[label] = [same(back), back.dtypes == want.dtypes]
        """
    )
    for label in ("pa.array", "pa.chunked_array", "pa.record_batch", "pa.table"):
        values, dtypes = got[label]
        assert values, f"{label}: the values pyarrow read are not the bank's"
        assert dtypes, f"{label}: the dtypes changed on the way through pyarrow"


def test_a_struct_pyarrow_has_read_is_spent() -> None:
    """Exporting hands the buffers to pyarrow, so a second read refuses
    rather than give them out twice."""
    got = _child(
        """
        out = fresh()
        pa.array(out)
        try:
            pa.array(out)
            result["second"] = "read twice"
        except ValueError as e:
            result["second"] = str(e)
        result["names"] = [f.name for f in pa.array(fresh()).type]
        """
    )
    assert "already been exported" in got["second"], got["second"]
    assert got["names"][:2] == ["pred_y", "resid_y"] and "coef" in got["names"], got["names"]


def test_a_pyarrow_reader_streams_into_a_bank() -> None:
    """A ``RecordBatchReader`` through ``pl.scan_arrow_c_stream``, batches of
    37 rows, into both streaming paths: the numbers are the whole frame's."""
    got = _child(
        """
        table = df.to_arrow()

        def reader():
            return pa.RecordBatchReader.from_batches(
                table.schema, table.to_batches(max_chunksize=37)
            )

        lazy = pl.scan_arrow_c_stream(reader()).online.fit_predict([SPEC]).collect()
        result["lf.online.fit_predict"] = same(lazy["m"].struct.unnest(), cadence_too=False)
        chunks = po.ModelBank([SPEC]).fit_predict_batches(
            pl.scan_arrow_c_stream(reader()), chunk_rows=23
        )
        batched = pl.concat(list(chunks))
        result["fit_predict_batches"] = same(batched["m"].struct.unnest(), cadence_too=False)
        """
    )
    assert got["lf.online.fit_predict"], "the query path over a pyarrow reader moved a number"
    assert got["fit_predict_batches"], "the batch path over a pyarrow reader moved a number"


def test_a_reused_pyarrow_reader_plan_warns() -> None:
    """A reader is single-use, like any Arrow C stream: a second fit over the
    same plan sees no rows, and ``ConsumedSourceWarning`` says so, as its
    message promises for a pyarrow reader."""
    got = _child(
        """
        table = df.to_arrow()
        reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
        lf = pl.scan_arrow_c_stream(reader)
        bank = po.ModelBank([SPEC])
        with warnings.catch_warnings(record=True) as first:
            warnings.simplefilter("always")
            bank.fit(lf)
        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            bank.fit(lf)
        result["first"] = [w.category.__name__ for w in first]
        result["second"] = [w.category.__name__ for w in second]
        """
    )
    assert "ConsumedSourceWarning" not in got["first"], got["first"]
    assert "ConsumedSourceWarning" in got["second"], got["second"]
