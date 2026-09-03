"""E33: the bank as a polars source -- `lf.online.fit_predict(specs)`.

A LazyFrame in, a LazyFrame out, streamed through a fresh bank when the plan
runs, so a query with the bank in it stays O(chunk); the expression plugin in
the same position is O(data) in either engine (docs/PERFORMANCE.md section
11). Held here to the numbers `ModelBank` and `po.run` give on the same rows,
through both engines, every pushdown polars applies to a Python source, a
streaming sink, and the eager and typed spellings.
"""

from __future__ import annotations

import math
import time

import numpy as np
import polars as pl
import pytest
from polars.io.plugins import register_io_source

import polars_online as po


def _frame(n=5000, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    return pl.DataFrame(
        {
            "t": np.arange(float(n)),
            "x0": x0,
            "y": 2 * x0 + 0.1 * rng.standard_normal(n),
            "g": np.where(np.arange(n) % 2 == 0, "a", "b"),
        }
    )


def _spec(**kw):
    d = dict(
        targets=["y"],
        features=["x0"],
        clock="t",
        max_dclock=10.0,
        halflife=500.0,
        group="g",
        min_periods=20.0,
    )
    d.update(kw)
    return po.spec.ewridge("ridge", **d)


def _bank_loop(df: pl.DataFrame, chunk_rows: int, **kw) -> pl.DataFrame:
    """The reference: a bank fed the same chunks."""
    bank = po.ModelBank([_spec(**kw)])
    parts = [bank.fit_predict(df.slice(i, chunk_rows)) for i in range(0, df.height, chunk_rows)]
    return pl.concat(parts)


def _no_coef(df: pl.DataFrame) -> pl.DataFrame:
    """`coef` is a reporting cadence -- snapshotted on each chunk's last row --
    and so the one field allowed to differ between chunkings."""
    return df.with_columns(pl.col("ridge").struct.with_fields(pl.lit(None).alias("coef")))


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_the_plan_gives_the_banks_numbers(engine):
    df = _frame()
    plan = df.lazy().online.fit_predict([_spec()], chunk_rows=1000)
    assert isinstance(plan, pl.LazyFrame)
    assert plan.collect(engine=engine).equals(_bank_loop(df, 1000))
    # ... and the whole-frame bank's, up to coef's cadence.
    assert _no_coef(plan.collect(engine=engine)).equals(
        _no_coef(po.ModelBank([_spec()]).fit_predict(df))
    )


def test_chunk_rows_is_only_a_resource_knob():
    df = _frame(n=3000)
    want = _no_coef(po.ModelBank([_spec()]).fit_predict(df))
    for chunk_rows in (37, 1000, 100_000):
        got = df.lazy().online.fit_predict([_spec()], chunk_rows=chunk_rows).collect()
        assert _no_coef(got).equals(want), chunk_rows
    with pytest.raises(ValueError, match="chunk_rows must be at least 1"):
        df.lazy().online.fit_predict([_spec()], chunk_rows=0)


def test_the_plan_is_pure():
    """Every execution starts from the same state: a fresh bank per run."""
    df = _frame(n=2000)
    plan = df.lazy().online.fit_predict([_spec()])
    first = plan.collect()
    assert plan.collect().equals(first)
    assert plan.collect(engine="streaming").equals(first)
    assert pl.concat(plan.collect_batches(chunk_size=300)).equals(first)


def test_pushdowns_are_honoured_after_the_model():
    """Polars pushes a filter, a projection and a `head` into a Python source
    and does not apply them again afterwards; each must come out as it would
    from the collected frame -- the filter applied *after* the bank, so that a
    downstream filter never changes what the bank learns from."""
    df = _frame()
    ref = _bank_loop(df, 1000)
    plan = df.lazy().online.fit_predict([_spec()], chunk_rows=1000)
    n_eff = pl.col("ridge").struct.field("n_eff")
    cases = {
        "filter on the model's output": lambda lf: lf.filter(n_eff > 30.0),
        "filter on an input column": lambda lf: lf.filter(pl.col("g") == "a"),
        "filter matching nothing": lambda lf: lf.filter(pl.col("t") < 0),
        "select": lambda lf: lf.select("t", "ridge"),
        "select a struct field": lambda lf: lf.select(pl.col("ridge").struct.field("pred_y")),
        "head": lambda lf: lf.head(2500),
        "head across chunks": lambda lf: lf.head(1234),
        "head then filter": lambda lf: lf.head(100).filter(pl.col("g") == "a"),
        "filter then head": lambda lf: lf.filter(pl.col("g") == "a").head(100),
        "all three": lambda lf: lf.filter(n_eff > 30.0).select("g", "ridge").head(700),
    }
    for name, q in cases.items():
        want = q(ref.lazy()).collect()
        for engine in ("streaming", "in-memory"):
            got = q(plan).collect(engine=engine)
            if "head" in name:
                # The rows a `head` asks for are the bank's last chunk, so
                # `coef` -- reported on each chunk's last row -- lands on the
                # last row delivered; the reference's chunk ran on past it.
                got, want = _no_coef(got), _no_coef(want)
            assert got.equals(want), (name, engine)
    # A filter that means to change what the bank learns from goes before it.
    want = _bank_loop(df.filter(pl.col("g") == "a"), 1000)
    got = df.lazy().filter(pl.col("g") == "a").online.fit_predict([_spec()], chunk_rows=1000)
    assert got.collect().equals(want)


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_the_input_is_read_in_chunks_and_only_as_far_as_needed(engine):
    """The input is pulled a chunk at a time, and a plan that stops early
    stops reading: `head(10)` over 100 input batches requests a handful (the
    engine reads a few morsels ahead; 7 measured) and none after the plan is
    done, so the input query is torn down with the plan."""
    df = _frame(n=50000)
    calls: list[int] = []

    def source(with_columns, predicate, n_rows, batch_size):
        for i in range(0, df.height, 500):
            calls.append(i)
            yield df.slice(i, 500)

    plan = register_io_source(source, schema=df.schema).online.fit_predict(
        [_spec()], chunk_rows=500
    )
    assert plan.collect(engine=engine).equals(_bank_loop(df, 500))
    assert len(calls) == 100
    calls.clear()
    assert plan.head(10).collect(engine=engine).height == 10
    assert 0 < len(calls) < 20
    time.sleep(0.2)
    assert len(calls) < 20


@pytest.mark.filterwarnings("ignore::polars.exceptions.PolarsInefficientMapWarning")
def test_projection_reaches_the_input():
    """A column the query does not ask for and no spec reads is never read:
    here one computed by a UDF that counts its calls."""
    df = _frame(n=1000)
    calls: list[int] = []

    def expensive(v: float) -> float:
        calls.append(1)
        return v

    lf = df.lazy().with_columns(wide=pl.col("x0").map_elements(expensive, return_dtype=pl.Float64))
    plan = lf.online.fit_predict([_spec()], chunk_rows=1000)
    got = plan.select("t", "ridge").collect()
    assert got.equals(_bank_loop(df, 1000).select("t", "ridge"))
    assert calls == []
    assert plan.collect().columns == ["t", "x0", "y", "g", "wide", "ridge"]
    assert len(calls) == 1000


def test_sink_equals_run(tmp_path):
    df = _frame(n=20000)
    src = tmp_path / "in.parquet"
    df.write_parquet(src)
    ran = tmp_path / "run.parquet"
    po.run(input=src, output=ran, specs=[_spec()], chunk_rows=3000)
    sunk = tmp_path / "sink.parquet"
    pl.scan_parquet(src).online.fit_predict([_spec()], chunk_rows=3000).sink_parquet(
        sunk, engine="streaming"
    )
    assert pl.read_parquet(sunk).equals(pl.read_parquet(ran))
    # ... and composes with polars after the bank, in the streaming engine.
    out = (
        pl.scan_parquet(src)
        .online.fit_predict([_spec()], chunk_rows=3000)
        .filter(pl.col("ridge").struct.field("n_eff") > 30.0)
        .group_by("g")
        .agg(pl.col("ridge").struct.field("resid_y").abs().mean().alias("mae"))
        .sort("g")
        .collect(engine="streaming")
    )
    want = (
        pl.read_parquet(ran)
        .filter(pl.col("ridge").struct.field("n_eff") > 30.0)
        .group_by("g")
        .agg(pl.col("ridge").struct.field("resid_y").abs().mean().alias("mae"))
        .sort("g")
    )
    assert out["g"].to_list() == want["g"].to_list()
    # A mean's summation order is the engine's; the rows it sums are ours.
    assert np.allclose(out["mae"].to_numpy(), want["mae"].to_numpy(), rtol=1e-12)


def test_load_state_resumes_the_bank(tmp_path):
    df = _frame(n=4000)
    head, tail = df.head(2500), df.tail(1500)
    bank = po.ModelBank([_spec()])
    bank.fit_predict(head)
    state = tmp_path / "bank.state"
    bank.save(state)
    want = bank.fit_predict(tail)
    plan = tail.lazy().online.fit_predict(load_state=state)
    assert plan.collect().equals(want)
    assert plan.collect().equals(want)  # the plan carries the state; each run starts from it
    assert tail.lazy().online.fit_predict([_spec()], load_state=state).collect().equals(want)
    assert tail.online.fit_predict(load_state=state).equals(want)
    with pytest.raises(ValueError, match="online.fit_predict needs specs, or load_state="):
        tail.lazy().online.fit_predict()


def test_predict_scores_the_bank_as_it_stands(tmp_path):
    df = _frame(n=4000)
    bank = po.ModelBank([_spec()])
    bank.fit_predict(df.head(3000))
    state = tmp_path / "bank.state"
    bank.save(state)
    today = df.tail(1000)
    want = bank.predict(today)
    for name, b in {"bank": bank, "path": state, "str": str(state)}.items():
        assert today.lazy().online.predict(b).collect().equals(want), name
        assert today.lazy().online.predict(b).collect(engine="streaming").equals(want), name
        assert today.online.predict(b).equals(want), name
        # `predict` reports coef on each chunk's last row, as fit_predict does.
        chunked = today.lazy().online.predict(b, chunk_rows=300).collect()
        assert _no_coef(chunked).equals(_no_coef(want)), name
    assert bank.rows_seen() == 3000  # nothing learned
    # The target is optional when scoring.
    no_target = today.lazy().online.predict(bank).select("t")
    assert no_target.collect().equals(today.select("t"))
    assert (
        today.drop("y").lazy().online.predict(bank).collect().equals(bank.predict(today.drop("y")))
    )


def test_the_eager_and_typed_spellings_are_the_bank():
    df = _frame(n=1500)
    want = po.ModelBank([_spec()]).fit_predict(df)
    assert df.online.fit_predict([_spec()]).equals(want)
    assert po.fit_predict(df, [_spec()]).equals(want)
    assert po.fit_predict(df.lazy(), [_spec()]).collect().equals(want)
    assert isinstance(po.fit_predict(df.lazy(), [_spec()]), pl.LazyFrame)
    bank = po.ModelBank([_spec()])
    bank.fit_predict(df)
    assert po.predict(df, bank).equals(bank.predict(df))
    assert po.predict(df.lazy(), bank).collect().equals(bank.predict(df))
    with pytest.raises(TypeError, match="online.fit_predict takes a polars DataFrame or LazyFrame"):
        po.fit_predict(df.to_dict(), [_spec()])  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="online.predict takes a polars DataFrame or LazyFrame"):
        po.predict([1, 2], bank)  # type: ignore[call-overload]


def test_an_empty_plan_has_the_schema():
    df = _frame(n=100)
    for plan in (df.clear().lazy(), df.lazy().filter(pl.lit(False))):
        out = plan.online.fit_predict([_spec()]).collect()
        assert out.height == 0
        assert out.schema == po.ModelBank([_spec()]).fit_predict(df).schema


def test_errors_name_the_problem():
    df = _frame(n=100)
    # A spec naming a column the plan lacks, or one of the wrong dtype, is
    # refused while the plan is built, the way polars reports its own schema
    # errors: the output schema comes from the bank run on no rows.
    with pytest.raises(ValueError, match=r'feature column "nope" not found'):
        df.lazy().online.fit_predict([_spec(features=["nope"])])
    bad = df.with_columns(pl.col("x0").cast(pl.String))
    with pytest.raises(ValueError, match="must be numeric"):
        bad.lazy().online.fit_predict([_spec()])
    # An error the rows raise surfaces as polars' ComputeError, carrying the
    # bank's message: here a clock that runs backwards.
    plan = df.reverse().lazy().online.fit_predict([_spec(on_clock_reset="error")])
    with pytest.raises(pl.exceptions.ComputeError, match="clock"):
        plan.collect()


def test_chunks_arrive_in_stream_order():
    """`maintain_order` holds through the plan: a shuffled input sorted in the
    plan reaches the bank sorted, and so does a per-group stream."""
    df = _frame(n=3000)
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=1)
    plan = shuffled.lazy().sort("t").online.fit_predict([_spec()], chunk_rows=400)
    assert plan.collect(engine="streaming").equals(_bank_loop(df, 400))
    n = plan.select(pl.col("ridge").struct.field("n_eff")).collect()["n_eff"]
    assert math.isclose(n.max(), _bank_loop(df, 400)["ridge"].struct.field("n_eff").max())


# --- E35: the state out of a streamed plan -- `save_state=` (docs/STATE-WORKFLOW.md).


def _bank_after(df: pl.DataFrame, **kw) -> bytes:
    """The reference: what a bank fed these rows saves."""
    bank = po.ModelBank([_spec(**kw)])
    bank.fit_predict(df)
    return bank.save_bytes()


def test_save_state_is_the_banks_state_after_the_stream(tmp_path):
    """C1: the plan writes, when it ends, the state a bank fed the same rows
    saves -- byte for byte, whatever the chunking, through either engine, and
    the same bytes `po.run(save_state=)` writes; so do the eager and typed
    spellings. Building the plan writes nothing."""
    df = _frame(n=4000)
    want = _bank_after(df)
    state = tmp_path / "bank.state"
    plan = df.lazy().online.fit_predict([_spec()], save_state=state, chunk_rows=700)
    assert not state.exists()
    for engine in ("streaming", "in-memory"):
        assert plan.collect(engine=engine).equals(_bank_loop(df, 700))
        assert state.read_bytes() == want, engine
        state.unlink()
    src = tmp_path / "in.parquet"
    df.write_parquet(src)
    po.run(input=src, output=tmp_path / "out.parquet", specs=[_spec()], save_state=state)
    assert state.read_bytes() == want
    for spell in (
        lambda: df.online.fit_predict([_spec()], save_state=state),
        lambda: po.fit_predict(df, [_spec()], save_state=state),
        lambda: po.fit_predict(df.lazy(), [_spec()], save_state=state).collect(),
        lambda: po.fit_predict(df.lazy(), [_spec()], save_state=str(state)).collect(),
    ):
        state.unlink()
        assert spell().equals(po.ModelBank([_spec()]).fit_predict(df))
        assert state.read_bytes() == want


def test_save_state_follows_head(tmp_path):
    """C2, R4: `head(n)` feeds the bank the first `n` rows and no more, so the
    state written is the state after `n` rows -- the rows the caller got --
    and not after the chunk they ended in. A `head` polars cannot push into
    the source (after a `sort`) runs the whole stream, and writes its state."""
    df = _frame(n=4000)
    state = tmp_path / "bank.state"
    plan = df.lazy().online.fit_predict([_spec()], save_state=state, chunk_rows=700)
    for n in (1, 699, 700, 701, 2345, 4000, 5000):
        got = plan.head(n).collect()
        assert got.height == min(n, 4000)
        assert _no_coef(got).equals(_no_coef(_bank_loop(df.head(n), 700)))
        assert state.read_bytes() == _bank_after(df.head(n)), n
    plan.sort("t", descending=True).head(5).collect()
    assert state.read_bytes() == _bank_after(df)


def test_a_plan_used_twice_in_one_query_writes_the_same_state_twice(tmp_path):
    """C3, R2: polars runs a Python source once per use of the plan in a
    query, concurrently, and does not share the runs (no CSE). Each run ends
    in the same state, so each write is the same bytes and the file is whole
    whichever finished last; no temporary is left behind."""
    df = _frame(n=4000)
    want = _bank_after(df)
    state = tmp_path / "bank.state"
    plan = df.lazy().online.fit_predict([_spec()], save_state=state, chunk_rows=500)
    ref = _bank_loop(df, 500)
    assert pl.concat([plan, plan]).collect().equals(pl.concat([ref, ref]))
    assert state.read_bytes() == want
    state.unlink()
    joined = plan.join(plan.select("t", "ridge"), on="t").collect()
    assert joined.height == 4000
    assert state.read_bytes() == want
    state.unlink()
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pl.collect_all(
        [plan.sink_parquet(a, lazy=True), plan.select("t", "ridge").sink_parquet(b, lazy=True)]
    )
    assert pl.read_parquet(a).equals(ref)
    assert pl.read_parquet(b).equals(ref.select("t", "ridge"))
    assert state.read_bytes() == want
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.parquet", "b.parquet", "bank.state"]


def test_a_run_that_does_not_reach_the_end_writes_nothing(tmp_path):
    """R1: the state is written by a run that fed the bank its last row and
    by nothing else -- not by one the caller abandoned (polars closes the
    source when it drops the plan), not by one the bank ended with an error."""
    df = _frame(n=40000)
    state = tmp_path / "bank.state"
    plan = df.lazy().online.fit_predict([_spec()], save_state=state, chunk_rows=500)
    batches = plan.collect_batches(chunk_size=500)
    assert next(batches).height == 500
    del batches, plan  # the engine read a few chunks ahead, out of 80
    time.sleep(0.2)
    assert not state.exists()
    plan = (
        df.reverse()
        .lazy()
        .online.fit_predict([_spec(on_clock_reset="error")], save_state=state, chunk_rows=500)
    )
    with pytest.raises(pl.exceptions.ComputeError, match="clock"):
        plan.collect()
    assert not state.exists()
    # The known gap (R6): a node *after* the bank failing does not stop the
    # bank -- polars drains a Python source before it raises -- so the state
    # after the whole stream is written although the query failed. `po.run`
    # saves only once its output is committed.
    plan = df.lazy().online.fit_predict([_spec()], save_state=state, chunk_rows=500)
    with pytest.raises(pl.exceptions.InvalidOperationError):
        plan.with_columns(pl.col("g").cast(pl.Int64, strict=True)).collect()
    assert state.read_bytes() == _bank_after(df)


def test_load_and_save_the_same_path_resumes_in_place(tmp_path):
    """C8: `load_state=p, save_state=p` over successive batches of the stream
    is one continuous stream -- the rows the bank gives, and the state it
    ends in -- and a plan built between two batches carries the state it was
    built from (R3): the file changing under it does not change its frame."""
    df = _frame(n=4000)
    state = tmp_path / "bank.state"
    parts = [df.slice(0, 900), df.slice(900, 1300), df.slice(2200, 1800)]
    got = []
    for i, part in enumerate(parts):
        plan = part.lazy().online.fit_predict(
            [_spec()], load_state=state if i else None, save_state=state, chunk_rows=500
        )
        got.append(plan.collect())
    assert _no_coef(pl.concat(got)).equals(_no_coef(_bank_loop(df, 500)))
    assert state.read_bytes() == _bank_after(df)
    # The plan carries the loaded state, so collecting it twice with the
    # file rewritten in between gives the same frame; the same for `predict`.
    resume = parts[2].lazy().online.fit_predict(load_state=state, chunk_rows=500)
    score = parts[2].lazy().online.predict(state, chunk_rows=500)
    first, scored = resume.collect(), score.collect()
    parts[0].lazy().online.fit_predict([_spec()], save_state=state).collect()
    assert resume.collect().equals(first)
    assert score.collect().equals(scored)
    assert not parts[2].lazy().online.predict(state).collect().equals(scored)
    # ... and `load_state=p, save_state=p` used twice in one query resumes
    # from the file's state twice, not from what the first run wrote.
    plan = (
        parts[1]
        .lazy()
        .online.fit_predict([_spec()], load_state=state, save_state=state, chunk_rows=500)
    )
    pl.concat([plan, plan]).collect()
    assert state.read_bytes() == _bank_after(pl.concat(parts[:2]))


def test_save_state_is_checked_when_the_plan_is_built(tmp_path):
    df = _frame(n=100)
    with pytest.raises(FileNotFoundError, match="save_state: .* is not a directory"):
        df.lazy().online.fit_predict([_spec()], save_state=tmp_path / "nope" / "bank.state")
    with pytest.raises(FileNotFoundError, match="save_state: .* is not a directory"):
        df.online.fit_predict([_spec()], save_state=tmp_path / "nope" / "bank.state")
    with pytest.raises(FileNotFoundError):
        df.lazy().online.fit_predict(load_state=tmp_path / "nope.state")
    with pytest.raises(FileNotFoundError):
        df.lazy().online.predict(tmp_path / "nope.state")
    # `predict` moves no state and so has nothing to save.
    with pytest.raises(TypeError, match="save_state"):
        df.lazy().online.predict(tmp_path / "nope.state", save_state=tmp_path / "s")  # type: ignore[call-arg]
