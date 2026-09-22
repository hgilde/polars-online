"""The bank's output over the Arrow PyCapsule interface (docs/PLAN.md task 86).

``ModelBank.fit_predict`` hands its struct columns to Python as pyo3-polars
``PySeries``, which reaches py-polars' private ``_export``/``_import`` -- the
reason this package carries a polars floor and the reason that interface
promises no stability. The Arrow pair hands the same values over
``__arrow_c_array__``, which is public and standardised, so any Arrow consumer
can read them.

These hold the two paths to each other. The values must be identical, nulls
and the nested ``coef`` list included, because it is the same computation:
only the way out differs.
"""

from __future__ import annotations

import polars as pl
import pytest

import polars_online as po

SPEC = {
    "name": "m",
    "model": {"type": "ew_ridge", "ridge": 1e-6, "max_rows_between_solves": 1},
    "targets": ["y"],
    "features": ["x0", "x1"],
    "clock": "t",
    "halflife": 60.0,
    "max_dclock": 30.0,
    "weight": "w",
    "group": "g",
    "min_periods": 5.0,
}


def frame(n: int = 200) -> pl.DataFrame:
    """Two groups, nulls in a feature and in the target."""
    return pl.DataFrame(
        {
            "g": [f"g{i % 2}" for i in range(n)],
            "t": [float(i // 2 + 1) for i in range(n)],
            "x0": [None if i % 17 == 5 else (i % 7) * 0.3 - 1.0 for i in range(n)],
            "x1": [(i % 5) * 0.25 - 0.5 for i in range(n)],
            "y": [None if i % 23 == 7 else (i % 11) * 0.1 for i in range(n)],
            "w": [1.0 + (i % 3) * 0.5 for i in range(n)],
        }
    )


def same(got: pl.Series, want: pl.Series) -> None:
    """Every struct field identical, nulls counted rather than skipped."""
    assert got.dtype == want.dtype
    assert len(got) == len(want)
    g, w = got.rename(want.name).struct.unnest(), want.struct.unnest()
    assert list(g.columns) == list(w.columns)
    assert g.null_count().rows() == w.null_count().rows()
    for c in w.columns:
        assert g[c].equals(w[c], null_equal=True), c


def test_the_arrow_output_is_the_polars_output() -> None:
    df = frame()
    want = po.ModelBank([SPEC]).fit_predict(df)["m"]
    outs = po.ModelBank([SPEC]).fit_predict_arrow(df)
    assert len(outs) == 1
    same(pl.Series(outs[0]), want)


def test_the_capsule_carries_the_spec_name() -> None:
    """`pl.Series` does not take a name from the caller here, so the field's
    own name is what arrives -- and it must be the spec's."""
    outs = po.ModelBank([SPEC]).fit_predict_arrow(frame(40))
    assert outs[0].name == "m"
    assert pl.Series(outs[0]).name == "m"


def test_predict_arrow_is_predict() -> None:
    df = frame(160)
    by_polars, by_arrow = po.ModelBank([SPEC]), po.ModelBank([SPEC])
    by_polars.fit_predict(df)
    by_arrow.fit_predict_arrow(df)
    want = by_polars.predict(df)["m"]
    outs = by_arrow.predict_arrow(df)
    same(pl.Series(outs[0]), want)


def test_a_struct_exports_once() -> None:
    """The interface hands its buffers to the consumer, so a second export
    would give away what has already been given. It refuses instead."""
    outs = po.ModelBank([SPEC]).fit_predict_arrow(frame(20))
    pl.Series(outs[0])
    with pytest.raises(ValueError, match="already been exported"):
        pl.Series(outs[0])


def test_one_struct_per_spec_in_spec_order() -> None:
    second = {**SPEC, "name": "n", "halflife": 10.0}
    df = frame(80)
    want = po.ModelBank([SPEC, second]).fit_predict(df)
    outs = po.ModelBank([SPEC, second]).fit_predict_arrow(df)
    assert [o.name for o in outs] == ["m", "n"]
    for o in outs:
        same(pl.Series(o), want[o.name])


def test_the_nested_coef_list_survives() -> None:
    """A null list and a null inside a list are different things; the capsule
    must preserve both, which a flat comparison would not catch."""
    df = frame(120)
    want = po.ModelBank([SPEC]).fit_predict(df)["m"].struct.unnest()
    outs = po.ModelBank([SPEC]).fit_predict_arrow(df)
    got = pl.Series(outs[0]).rename("m").struct.unnest()
    assert got["coef"].dtype == pl.List(pl.Float64)
    assert got["coef"].null_count() == want["coef"].null_count()
    assert got["coef"].equals(want["coef"], null_equal=True)


def test_an_empty_frame_gives_an_empty_struct_of_the_right_dtype() -> None:
    """Nothing to feed is not an error, and the schema is still the spec's."""
    df = frame()
    out = po.ModelBank([SPEC]).fit_predict_arrow(df.clear())
    s = pl.Series(out[0])
    assert len(s) == 0
    assert s.dtype == po.ModelBank([SPEC]).fit_predict(df)["m"].dtype


def test_a_multi_chunk_frame_equals_the_rechunked_one() -> None:
    """A batch from a scan can arrive as several chunks; the adapter reads them
    as one array, and the numbers must not know the difference."""
    df = frame()
    mc = pl.concat([df.slice(0, 50), df.slice(50)], rechunk=False)
    assert mc.n_chunks() > 1
    got = pl.Series(po.ModelBank([SPEC]).fit_predict_arrow(mc)[0])
    same(got, po.ModelBank([SPEC]).fit_predict(df.rechunk())["m"])


def test_predict_arrow_on_a_bank_that_has_seen_nothing_is_null_throughout() -> None:
    """A group the bank has never learned from scores as null, as ``predict``
    documents -- over the Arrow path too."""
    out = po.ModelBank([SPEC]).predict_arrow(frame())
    s = pl.Series(out[0]).struct.unnest()
    assert s["pred_y"].null_count() == len(s)


def test_the_arrow_pair_refuses_a_lazyframe_by_name() -> None:
    """The type check names the method the caller used, not its polars twin."""
    with pytest.raises(TypeError, match="fit_predict_arrow takes a DataFrame"):
        po.ModelBank([SPEC]).fit_predict_arrow(frame().lazy())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="predict_arrow takes a DataFrame"):
        po.ModelBank([SPEC]).predict_arrow(frame().lazy())  # type: ignore[arg-type]


# --- consumers that are not polars -------------------------------------------
#
# The README says these structs can go to another Arrow consumer.
# ``docs/ARROW-SOURCES.md`` §3 flagged that as an untested claim, and testing it
# showed the duckdb half was wrong: DuckDB reads ``__arrow_c_stream__``, which a
# ``Series`` carries and an ``ArrowStruct`` does not. These pin the real
# behaviour so the README cannot drift back.


def test_the_struct_carries_the_array_dunder_and_not_the_stream_one() -> None:
    """Which dunder a struct carries is what decides the consumers it can reach,
    so it is pinned here rather than inferred from a failure somewhere else."""
    out = po.ModelBank([SPEC]).fit_predict_arrow(frame(20))[0]
    assert hasattr(out, "__arrow_c_array__")
    assert not hasattr(out, "__arrow_c_stream__")


def test_duckdb_refuses_the_struct_directly() -> None:
    """Measured on duckdb 1.5.5. "Hand them straight to duckdb" was wrong, and
    this is the refusal that says so. If DuckDB ever grows array-interface
    support this fails, and the README can promise the direct path again."""
    duckdb = pytest.importorskip("duckdb")
    out = po.ModelBank([SPEC]).fit_predict_arrow(frame(20))[0]
    with pytest.raises(duckdb.InvalidInputException, match="not an accepted Arrow Object"):
        duckdb.from_arrow(out)


def test_duckdb_takes_the_same_values_through_a_series() -> None:
    """The route the README now gives. A ``Series`` carries the stream dunder,
    so DuckDB accepts it, and the values that arrive are the bank's own."""
    duckdb = pytest.importorskip("duckdb")
    df = frame(60)
    want = po.ModelBank([SPEC]).fit_predict(df)["m"].struct.unnest()
    got = pl.Series(po.ModelBank([SPEC]).fit_predict_arrow(df)[0])

    rel = duckdb.from_arrow(got)
    assert "pred_y" in rel.columns, "the struct fields should arrive as columns"
    rows = rel.project("pred_y").fetchall()
    assert len(rows) == len(want)
    assert pl.Series("pred_y", [r[0] for r in rows]).equals(want["pred_y"], null_equal=True)


def test_a_struct_handed_to_duckdb_is_still_spent() -> None:
    """The export-once contract is the capsule's, not polars'. Reading through a
    ``Series`` on the way to DuckDB consumes the struct exactly as any other
    consumer would, so a second read refuses rather than double-free."""
    pytest.importorskip("duckdb")
    out = po.ModelBank([SPEC]).fit_predict_arrow(frame(20))[0]
    pl.Series(out)
    with pytest.raises(ValueError, match="already been exported"):
        pl.Series(out)
