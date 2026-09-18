"""A plan that read a spent Arrow stream is reported, not passed off as a fit.

An Arrow C stream is consumed once -- the PyCapsule specification says a
capsule "can only be consumed once" -- so a ``LazyFrame`` from
``pl.scan_arrow_c_stream(...)`` over a DuckDB relation or a pyarrow reader
works the first time it is collected and yields **nothing** every time after,
with no error from polars (checked: 1,000 rows, then 0, silently). A bank fed
that plan twice learns from every row and then from none, and the second run
leaves a state that looks finished and is empty.

The discriminator is not "a plan was reused": re-collecting a
``scan_parquet`` is legitimate and gives its rows every time. It is not
"PYTHON SCAN" either -- this package's own plan form carries that marker and
*is* reusable. It is the pair: the source is a Python scan **and** the run saw
no rows. An in-memory ``.lazy()`` shows ``DF [...]`` and a ``scan_parquet``
shows a parquet scan, so neither can trip it, and an empty one of either stays
quiet -- which is what keeps ``fit`` over a genuinely empty plan an ordinary
run rather than a warning.

No DuckDB or pyarrow here: ``register_io_source`` builds a source with the
same single-use behaviour, which is the shape under test.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest
from polars.io.plugins import register_io_source

import polars_online as po

SPEC = po.spec.ewridge(
    "m", targets=["y"], features=["x0"], clock="t", halflife=10.0, max_dclock=5.0, min_periods=1.0
)


def _frame(n: int = 8) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [float(i) for i in range(n)],
            "x0": [0.1 * i for i in range(n)],
            "y": [0.2 * i for i in range(n)],
        }
    )


def _single_use(frame: pl.DataFrame | None = None) -> pl.LazyFrame:
    """A plan that yields its rows once and nothing after -- what a consumed
    Arrow C stream does, without needing one. The ``spent`` flag lives in the
    closure, so it survives between collects of the same ``LazyFrame`` exactly
    as a producer's handed-away buffers do."""
    frame = _frame() if frame is None else frame
    spent: list[bool] = []

    def io_source(with_columns, predicate, n_rows, batch_size):  # noqa: ANN001, ANN202
        if spent:
            return
        spent.append(True)
        out = frame
        if n_rows is not None:
            out = out.head(n_rows)
        if predicate is not None:
            out = out.filter(predicate)
        if with_columns is not None:
            out = out.select(with_columns)
        yield out

    return register_io_source(io_source=io_source, schema=frame.schema)


def _bank() -> po.ModelBank:
    return po.ModelBank([SPEC])


# --- the hazard itself -------------------------------------------------------


def test_the_first_run_over_a_single_use_plan_is_quiet():
    """It has rows, so there is nothing to report -- the guard must not fire
    merely because the source is a Python scan."""
    lf = _single_use()
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.ConsumedSourceWarning)
        _bank().fit(lf)


def test_the_second_run_over_a_single_use_plan_warns():
    """The run the user actually loses: no rows, no error from polars, a bank
    that learned nothing."""
    lf = _single_use()
    bank = _bank()
    bank.fit(lf)
    with pytest.warns(po.ConsumedSourceWarning, match="consumed once"):
        bank.fit(lf)


def test_the_bank_really_did_learn_nothing_on_the_second_run():
    """The warning is not decorative: the state after the second run is the
    state after the first, because no row reached the bank."""
    lf = _single_use()
    first, second = _bank(), _bank()
    first.fit(lf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", po.ConsumedSourceWarning)
        second.fit(lf)
    assert second.coef("m").height == 0 or second.coef("m")["value"].is_null().all(), (
        "the second bank saw rows, so this test no longer reproduces the hazard"
    )
    assert first.coef("m").height > 0, "the first bank should have fit"


def test_fit_predict_batches_warns_too():
    lf = _single_use()
    bank = _bank()
    list(bank.fit_predict_batches(lf))
    with pytest.warns(po.ConsumedSourceWarning):
        list(bank.fit_predict_batches(lf))


def test_the_plan_form_warns():
    """``lf.online.fit_predict`` consumes the same way, so it reports the same
    way -- collected twice, the second gives an empty frame."""
    lf = _single_use()
    plan = lf.online.fit_predict([SPEC])
    plan.collect()
    with pytest.warns(po.ConsumedSourceWarning):
        out = plan.collect()
    assert out.height == 0


# --- what must stay quiet ----------------------------------------------------


def test_an_empty_in_memory_plan_does_not_warn():
    """``.lazy()`` shows ``DF [...]``, not a Python scan. An empty frame is an
    ordinary run, and ``fit`` over one stays an ordinary run."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.ConsumedSourceWarning)
        _bank().fit(_frame().clear().lazy())


def test_an_empty_parquet_plan_does_not_warn(tmp_path):
    """A parquet scan is reusable and re-collectable; empty is just empty."""
    path = tmp_path / "empty.parquet"
    _frame().clear().write_parquet(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.ConsumedSourceWarning)
        _bank().fit(pl.scan_parquet(path))


def test_a_parquet_plan_is_reusable_and_stays_quiet(tmp_path):
    """The false positive a blanket "warn on reuse" guard would produce."""
    path = tmp_path / "rows.parquet"
    _frame().write_parquet(path)
    lf = pl.scan_parquet(path)
    bank = _bank()
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.ConsumedSourceWarning)
        bank.fit(lf)
        bank.fit(lf)


def test_an_empty_frame_is_not_inspected():
    """A ``DataFrame`` is not a plan; there is no source to have spent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.ConsumedSourceWarning)
        _bank().fit(_frame().clear())


def test_an_empty_iterator_is_not_inspected():
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.ConsumedSourceWarning)
        _bank().fit(iter([]))


def test_a_head_zero_on_the_plan_form_does_not_warn():
    """``head(0)`` pushed into the scan means no rows is what the query asked
    for, not a stream that is spent."""
    lf = _single_use()
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.ConsumedSourceWarning)
        out = lf.online.fit_predict([SPEC]).head(0).collect()
    assert out.height == 0


# --- the property the guard rests on -----------------------------------------


def test_explain_does_not_consume_the_stream():
    """The guard reads ``explain`` on the caller's plan. If that consumed the
    stream, the guard would cause the very bug it reports."""
    lf = _single_use()
    lf.explain(optimized=False)
    lf.explain(optimized=True)
    assert lf.collect().height == 8, "explain consumed the stream"


def test_a_python_scan_with_rows_is_recognised_and_a_parquet_scan_is_not(tmp_path):
    """The discriminator itself, on the plan shapes it must separate."""
    from polars_online._frame import _is_python_scan

    path = tmp_path / "rows.parquet"
    _frame().write_parquet(path)
    assert _is_python_scan(_single_use())
    assert _is_python_scan(_frame().lazy().online.fit_predict([SPEC])), (
        "this package's own plan form is a Python scan -- which is why a zero-row "
        "run, not the marker alone, is what the guard keys on"
    )
    assert not _is_python_scan(_frame().lazy())
    assert not _is_python_scan(pl.scan_parquet(path))


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("PYTHON SCAN []", id="polars-1.x"),
        pytest.param("PYTHON[polars-online] SCAN []", id="polars-2.0-named"),
    ],
)
def test_both_spellings_of_a_python_scan_are_recognised(line):
    """polars 2.0 renders a source given an ``explain_name`` as
    ``PYTHON[<name>] SCAN``, and ``_explain_kwargs`` names this package's own
    plan form ``polars-online``. A literal ``"PYTHON SCAN" in text`` missed
    that, and the release run's 2.0 leg failed on it.

    Measured on 2.0.0-rc.1: ``scan_arrow_c_stream`` over a polars frame *and*
    over a DuckDB relation both still read ``PYTHON SCAN``, unnamed, so the
    spent-stream guard went on working there and only our own plan form was
    misread. Pinned as text so both spellings hold without a 2.0 install."""
    from polars_online._frame import _PYTHON_SCAN

    assert _PYTHON_SCAN.search(line)


@pytest.mark.parametrize(
    "line",
    [
        'DF ["t", "x0", "y"]; PROJECT */3 COLUMNS',
        "Parquet SCAN [/tmp/a.parquet]",
        "SCAN []",
    ],
)
def test_a_source_that_is_not_a_python_scan_is_not_matched(line):
    """The pattern must not widen into "anything with SCAN in it": an
    in-memory frame and a parquet scan are both reusable, and warning on them
    would be the false positive the guard is built to avoid."""
    from polars_online._frame import _PYTHON_SCAN

    assert not _PYTHON_SCAN.search(line)


def test_a_plan_that_cannot_be_explained_is_let_through(monkeypatch):
    """Best-effort: a plan the guard cannot read is never a reason to fail."""
    from polars_online import _frame as frame_mod

    def boom(self, **kwargs):  # noqa: ANN001, ANN202, ARG001
        raise RuntimeError("no plan for you")

    monkeypatch.setattr(pl.LazyFrame, "explain", boom)
    assert frame_mod._is_python_scan(_frame().lazy()) is False


# --- the warning as a warning ------------------------------------------------


def test_the_warning_is_a_user_warning_shown_by_default():
    assert issubclass(po.ConsumedSourceWarning, UserWarning)


def test_the_message_names_the_call_and_the_fix():
    lf = _single_use()
    bank = _bank()
    bank.fit(lf)
    with pytest.warns(po.ConsumedSourceWarning) as caught:
        bank.fit(lf)
    text = str(caught[0].message)
    assert "ModelBank.fit" in text, "the message should name what the caller called"
    assert "scan_arrow_c_stream" in text
    assert "inside the loop" in text, "the message should say how to fix it"
    assert "ConsumedSourceWarning" in text, "and how to silence it"


# --- the real thing, on DuckDB ------------------------------------------------


def _duck_rel(duckdb, n: int = 8):
    """A relation over ``n`` rows, ordered -- an online model learns in row
    order, so a relation feeding a bank wants an explicit ``ORDER BY``."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE ticks AS SELECT i::DOUBLE AS t, (i*0.1)::DOUBLE AS x0, "
        f"(i*0.2)::DOUBLE AS y FROM range({n}) AS r(i)"
    )
    return con, con.sql("SELECT t, x0, y FROM ticks ORDER BY t")


def test_the_hazard_reproduces_on_a_real_duckdb_relation():
    """Not a synthetic stand-in: the plan gives its rows once and nothing
    after, which is the bug this guard exists for."""
    duckdb = pytest.importorskip("duckdb")
    con, rel = _duck_rel(duckdb)
    lf = pl.scan_arrow_c_stream(rel)
    assert lf.collect().height == 8
    assert lf.collect().height == 0, "duckdb/polars no longer reproduce the hazard"
    con.close()


def test_a_bank_over_a_reused_duckdb_plan_warns():
    duckdb = pytest.importorskip("duckdb")
    con, rel = _duck_rel(duckdb)
    lf = pl.scan_arrow_c_stream(rel)
    bank = _bank()
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.ConsumedSourceWarning)
        bank.fit(lf)
    with pytest.warns(po.ConsumedSourceWarning):
        bank.fit(lf)
    con.close()


def test_rebuilding_the_plan_per_run_is_the_documented_fix():
    """What the warning tells the caller to do, checked as advertised."""
    duckdb = pytest.importorskip("duckdb")
    con, _ = _duck_rel(duckdb)
    bank = _bank()
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.ConsumedSourceWarning)
        for _ in range(2):
            rel = con.sql("SELECT t, x0, y FROM ticks ORDER BY t")
            bank.fit(pl.scan_arrow_c_stream(rel))
    con.close()


def test_a_duckdb_relation_itself_is_reusable_on_this_version():
    """Pins what DuckDB issue #17084 reported, because the document cites it.

    The issue says a relation's ``__arrow_c_stream__`` works once and then
    raises. On duckdb 1.5.5 it does **not** -- both calls succeed. So the
    single-use behaviour belongs to the captured *stream*, not to the
    relation, and if this test ever fails the claim in
    ``docs/ARROW-SOURCES.md`` §3 needs rewriting again.
    """
    duckdb = pytest.importorskip("duckdb")
    con, rel = _duck_rel(duckdb)
    rel.__arrow_c_stream__(None)
    rel.__arrow_c_stream__(None)
    con.close()


def test_the_pyarrow_free_path_is_the_one_the_docs_recommend():
    """``scan_arrow_c_stream`` rides the capsule interface; the reader APIs
    need pyarrow, which this project does not depend on."""
    duckdb = pytest.importorskip("duckdb")
    con, rel = _duck_rel(duckdb)
    assert pl.scan_arrow_c_stream(rel).collect().height == 8
    assert not hasattr(rel, "record_batch"), (
        "`record_batch` is back; the docs recipe can use it again"
    )
    con.close()


def test_the_warning_points_at_the_caller_not_the_library():
    lf = _single_use()
    bank = _bank()
    bank.fit(lf)
    with pytest.warns(po.ConsumedSourceWarning) as caught:
        bank.fit(lf)
    assert caught[0].filename == __file__, f"pointed at {caught[0].filename}, not the caller's file"
