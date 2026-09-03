"""E8 / E32: `polars_online.run` — the streaming runner from Python.

The same pipeline as the `online` CLI (`online_polars::run`), reading with
py-polars: a path in any of the four formats, a LazyFrame, a DataFrame or an
iterable of frames goes in and any format comes out, with memory O(state +
chunk) rather than O(data) and no process spawned. Every source and format is
held to the numbers `ModelBank` gives on the same rows.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po

# format name -> the extension `run` tells it from (one of each format's).
FORMATS = {"parquet": "parquet", "ipc": "arrow", "csv": "csv", "ndjson": "jsonl"}


def _frame(n=20000, seed=0) -> pl.DataFrame:
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


def _write(path, n=20000, seed=0):
    _frame(n, seed).write_parquet(path)


def _save(df: pl.DataFrame, path, fmt: str) -> None:
    getattr(df, f"write_{fmt}")(path)


def _load(path, fmt: str) -> pl.DataFrame:
    return getattr(pl, f"read_{fmt}")(path)


def _preds(df: pl.DataFrame) -> list:
    return df["ridge"].struct.field("pred_y").to_list()


def _bank_preds(df: pl.DataFrame, **kw) -> list:
    return _preds(po.ModelBank([_spec(**kw)]).fit_predict(df))


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


def test_runs_and_reports_stats(tmp_path):
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    _write(src)
    stats = po.run(input=src, output=dst, specs=[_spec()], chunk_rows=4000)
    assert stats == {"rows": 20000, "chunks": 5}
    out = pl.read_parquet(dst)
    assert out.height == 20000
    assert "ridge" in out.columns


def test_chunk_rows_is_only_a_resource_knob(tmp_path):
    src = tmp_path / "in.parquet"
    _write(src, n=5000)
    preds = []
    for chunk in (500, 5000):
        dst = tmp_path / f"out{chunk}.parquet"
        po.run(input=src, output=dst, specs=[_spec()], chunk_rows=chunk)
        preds.append(pl.read_parquet(dst)["ridge"].struct.field("pred_y").to_list())
    assert preds[0] == preds[1], "chunk size changed the numbers"


def test_save_and_resume(tmp_path):
    src = tmp_path / "in.parquet"
    _write(src, n=4000)
    df = pl.read_parquet(src)
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    df.slice(0, 2000).write_parquet(first)
    df.slice(2000, 2000).write_parquet(second)

    whole = tmp_path / "whole-out.parquet"
    po.run(input=src, output=whole, specs=[_spec()])
    expected = pl.read_parquet(whole)["ridge"].struct.field("pred_y").to_list()

    state = tmp_path / "bank.state"
    po.run(input=first, output=tmp_path / "o1.parquet", specs=[_spec()], save_state=state)
    po.run(input=second, output=tmp_path / "o2.parquet", specs=[_spec()], load_state=state)
    got = (
        pl.read_parquet(tmp_path / "o1.parquet")["ridge"].struct.field("pred_y").to_list()
        + pl.read_parquet(tmp_path / "o2.parquet")["ridge"].struct.field("pred_y").to_list()
    )
    assert got == expected, "resuming did not reproduce the unbroken run"


def test_matches_the_model_bank(tmp_path):
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    _write(src, n=3000)
    stats = po.run(input=src, output=dst, specs=[_spec()])
    assert stats == {"rows": 3000, "chunks": 1}, "the default chunk_rows is 100k"
    from_runner = pl.read_parquet(dst)["ridge"].struct.field("pred_y").to_list()
    from_bank = (
        po.ModelBank([_spec()])
        .fit_predict(pl.read_parquet(src))["ridge"]
        .struct.field("pred_y")
        .to_list()
    )
    assert from_runner == from_bank


def test_accepts_a_toml_config(tmp_path):
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    _write(src, n=2000)
    cfg = tmp_path / "bank.toml"
    cfg.write_text(
        f'input = "{src.as_posix()}"\n'
        f'output = "{dst.as_posix()}"\n'
        "\n[[specs]]\n"
        'name = "ridge"\n'
        'targets = ["y"]\n'
        'features = ["x0"]\n'
        "halflife = 500.0\n"
        "min_periods = 20.0\n"
        "\n[specs.model]\n"
        'type = "ew_ridge"\n'
    )
    stats = po.run(cfg)
    assert stats["rows"] == 2000
    assert dst.exists()


def test_keywords_override_the_config(tmp_path):
    src = tmp_path / "in.parquet"
    _write(src, n=1000)
    other = tmp_path / "other.parquet"
    _write(other, n=500, seed=1)
    cfg = {
        "input": str(src),
        "output": str(tmp_path / "unused.parquet"),
        "specs": [_spec(group=None)],
    }
    dst = tmp_path / "used.parquet"
    stats = po.run(cfg, input=other, output=dst)
    assert stats["rows"] == 500, "the keyword input should have won"
    assert dst.exists()
    assert not (tmp_path / "unused.parquet").exists()


def test_a_decimal_parquet_runs(tmp_path):
    """Prices in parquet are commonly `Decimal`, and small codes `UInt8`.
    Both used to be unreadable by this build -- `Decimal` panicked at the
    boundary with `activate 'dtype-decimal' feature` (IMPROVEMENTS U5) --
    which for the CLI meant a file it could not open at all."""
    src, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    _write(src, n=2000)
    pl.read_parquet(src).with_columns(
        price=pl.col("x0").cast(pl.Decimal(12, 4)),
        code=(pl.arange(0, 2000) % 3).cast(pl.UInt8),
    ).write_parquet(src)

    stats = po.run(input=src, output=out, specs=[_spec(features=["price", "code"])])
    assert stats["rows"] == 2000
    got = pl.read_parquet(out)
    assert got["ridge"].struct.field("pred_y").drop_nulls().len() > 0
    # And the same numbers the Float64 columns would have given.
    plain = pl.read_parquet(src).with_columns(
        price=pl.col("price").cast(pl.Float64), code=pl.col("code").cast(pl.Float64)
    )
    want = po.ModelBank([_spec(features=["price", "code"])]).fit_predict(plain)
    assert want["ridge"].equals(got["ridge"], null_equal=True)


def test_missing_specs_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one spec"):
        po.run(input=tmp_path / "x.parquet", output=tmp_path / "y.parquet")


def test_bad_spec_is_reported(tmp_path):
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    _write(src, n=100)
    bad = _spec()
    bad["features"] = ["does_not_exist"]
    with pytest.raises(ValueError, match="does_not_exist"):
        po.run(input=src, output=dst, specs=[bad])


def test_row_groups_need_not_align_with_chunk_rows(tmp_path):
    """A chunk that spans a parquet row-group boundary arrives as a multi-chunk
    frame; the bank's outputs are single-chunk. Writing the two together used
    to panic in the arrow record-batch writer ("RecordBatch requires all its
    arrays to have an equal number of rows") -- on every file whose row groups
    were not a multiple of `chunk_rows`, which is any file polars writes with
    its default 262144-row groups read with a smaller `chunk_rows`."""
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    n = 1000
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "t": np.arange(float(n)),
            "x0": x0,
            "y": 2 * x0 + 0.1 * rng.standard_normal(n),
            "g": np.where(np.arange(n) % 2 == 0, "a", "b"),
        }
    )
    df.write_parquet(src, row_group_size=50)
    stats = po.run(input=src, output=dst, specs=[_spec()], chunk_rows=80)
    assert stats == {"rows": n, "chunks": 13}
    out = pl.read_parquet(dst)
    assert out.drop("ridge").equals(df)
    from_bank = po.ModelBank([_spec()]).fit_predict(df)["ridge"].struct.field("pred_y")
    assert out["ridge"].struct.field("pred_y").to_list() == from_bank.to_list()


# ---------------------------------------------------------------------------
# E32: any source, any format (docs/ENHANCEMENTS.md).


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_every_input_format_gives_the_banks_numbers(tmp_path, fmt):
    df = _frame(n=3000)
    src, dst = tmp_path / f"in.{FORMATS[fmt]}", tmp_path / "out.parquet"
    _save(df, src, fmt)
    stats = po.run(input=src, output=dst, specs=[_spec()], chunk_rows=1000)
    assert stats == {"rows": 3000, "chunks": 3}
    out = pl.read_parquet(dst)
    assert _preds(out) == _bank_preds(df)
    # The input columns come through as they were, text formats included.
    assert out.drop("ridge").equals(df)


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_every_output_format_carries_the_banks_columns(tmp_path, fmt):
    """One chunk, so `ModelBank.fit_predict` on the whole frame is the very
    same computation -- `coef` included, which is also snapshotted on each
    chunk's last row."""
    df = _frame(n=3000)
    src, dst = tmp_path / "in.parquet", tmp_path / f"out.{FORMATS[fmt]}"
    df.write_parquet(src)
    po.run(input=src, output=dst, specs=[_spec(coef_every=7)], chunk_rows=3000)
    want = po.ModelBank([_spec(coef_every=7)]).fit_predict(df)["ridge"]
    out = _load(dst, fmt)
    if fmt == "csv":
        # CSV has no structs: `<spec>.<field>` columns, lists as JSON text.
        assert out.columns == [
            *df.columns,
            *(f"ridge.{f}" for f in ("pred_y", "resid_y", "n_eff", "coef")),
        ]
        assert out["ridge.pred_y"].to_list() == want.struct.field("pred_y").to_list()
        coef = out["ridge.coef"].str.json_decode(pl.List(pl.Float64))
        assert 300 < coef.null_count() < 3000, "rows between snapshots are null, not []"
        assert coef.to_list() == want.struct.field("coef").to_list(), "not bit-exact"
        return
    assert out.columns == [*df.columns, "ridge"]
    assert out.schema["ridge"].fields == want.dtype.fields
    assert out["ridge"].equals(want, null_equal=True)


def test_a_frame_a_plan_and_frames_equal_the_path(tmp_path):
    df = _frame(n=2500)
    src = tmp_path / "in.parquet"
    df.write_parquet(src)

    def frames():
        for i in range(0, 2500, 400):
            yield df.slice(i, 400)

    # source -> the chunks it arrives in: `chunk_rows` for what polars reads,
    # the frames themselves for what is handed in.
    sources = {
        "path": (src, 3),
        "plan": (pl.scan_parquet(src), 3),
        "frame": (df, 3),
        "list": ([df.slice(0, 1000), df.slice(1000)], 2),
        "generator": (frames(), 7),
    }
    want = _bank_preds(df)
    for name, (source, chunks) in sources.items():
        dst = tmp_path / f"{name}.parquet"
        stats = po.run(input=source, output=dst, specs=[_spec()], chunk_rows=1000)
        assert stats == {"rows": 2500, "chunks": chunks}, name
        assert _preds(pl.read_parquet(dst)) == want, name


@pytest.mark.filterwarnings("ignore::polars.exceptions.PolarsInefficientMapWarning")
def test_a_query_streams_through(tmp_path):
    """Any plan: here a filter and a Python UDF (deliberately -- it needs the
    GIL on polars' threads while the runner is inside Rust)."""
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    _write(src, n=2000)
    query = (
        pl.scan_parquet(src)
        .filter(pl.col("g") == "a")
        .with_columns(x2=pl.col("x0").map_elements(lambda v: 2.0 * v, return_dtype=pl.Float64))
    )
    kw = dict(features=["x2"], group=None)
    stats = po.run(input=query, output=dst, specs=[_spec(**kw)], chunk_rows=300)
    assert stats == {"rows": 1000, "chunks": 4}
    assert _preds(pl.read_parquet(dst)) == _bank_preds(query.collect(), **kw)


def test_progress_is_called_after_every_chunk(tmp_path):
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    _write(src, n=2500)
    calls = []
    po.run(
        input=src,
        output=dst,
        specs=[_spec()],
        chunk_rows=1000,
        progress=lambda r, c: calls.append((r, c)),
    )
    assert calls == [(1000, 1), (2000, 2), (2500, 3)]
    with pytest.raises(TypeError, match="callable"):
        po.run(input=src, output=dst, specs=[_spec()], progress="yes")  # type: ignore[arg-type]


class _Stop(Exception):
    pass


def test_raising_in_progress_stops_the_run_and_keeps_the_old_output(tmp_path):
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    _write(src, n=2500)
    dst.write_bytes(b"previous")

    def progress(rows, chunks):
        if chunks == 2:
            raise _Stop(rows)

    with pytest.raises(_Stop, match="2000"):
        po.run(input=src, output=dst, specs=[_spec()], chunk_rows=1000, progress=progress)
    assert dst.read_bytes() == b"previous", "a failed run must not publish"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.parquet", "out.parquet"], "temp left"


def test_raising_in_the_frames_surfaces_as_itself(tmp_path):
    df = _frame(n=1000)
    dst = tmp_path / "out.parquet"

    def frames():
        yield df.slice(0, 500)
        yield df.slice(500, 500 // 0)

    with pytest.raises(ZeroDivisionError):
        po.run(input=frames(), output=dst, specs=[_spec()])
    assert not dst.exists()


def test_non_frames_are_rejected(tmp_path):
    df = _frame(n=100)
    dst = tmp_path / "out.parquet"
    with pytest.raises(TypeError, match="must be polars DataFrames, got int"):
        po.run(input=[1, 2], output=dst, specs=[_spec()])
    with pytest.raises(TypeError, match="got dict"):
        po.run(input=[df, {"t": 1}], output=dst, specs=[_spec()])
    assert not dst.exists()
    with pytest.raises(ValueError, match="no frames"):
        po.run(input=[], output=dst, specs=[_spec()])
    with pytest.raises(ValueError, match="needs an input"):
        po.run(output=dst, specs=[_spec()])
    with pytest.raises(FileNotFoundError):
        po.run(input=tmp_path / "missing.parquet", output=dst, specs=[_spec()])
    with pytest.raises(TypeError, match="config must be a dict, a path to a TOML file, or None"):
        po.run(42, input=df, output=dst, specs=[_spec()])
    with pytest.raises(TypeError, match="progress must be callable, got str"):
        po.run(input=df, output=dst, specs=[_spec()], progress="tick")


def test_an_empty_input_writes_the_schema(tmp_path):
    df = _frame(n=100)
    for name, source in {"frame": df.clear(), "plan": df.lazy().filter(pl.lit(False))}.items():
        dst = tmp_path / f"{name}.parquet"
        assert po.run(input=source, output=dst, specs=[_spec()]) == {"rows": 0, "chunks": 0}
        out = pl.read_parquet(dst)
        assert out.height == 0 and out.columns == [*df.columns, "ridge"], name
        assert "pred_y" in [f.name for f in out.schema["ridge"].fields], name
    csv = tmp_path / "empty.csv"
    po.run(input=df.clear(), output=csv, specs=[_spec()])
    assert pl.read_csv(csv).columns == [
        *df.columns,
        "ridge.pred_y",
        "ridge.resid_y",
        "ridge.n_eff",
        "ridge.coef",
    ]


def test_keep_columns_selects_the_input(tmp_path):
    df = _frame(n=1000).with_columns(junk=pl.lit("x"))
    src = tmp_path / "in.parquet"
    df.write_parquet(src)
    want = _bank_preds(df.drop("junk"))
    for name, source in {"path": src, "frames": [df.slice(0, 600), df.slice(600)]}.items():
        dst = tmp_path / f"{name}.parquet"
        po.run(input=source, output=dst, specs=[_spec()], keep_columns=["t", "x0", "y", "g"])
        out = pl.read_parquet(dst)
        assert out.columns == ["t", "x0", "y", "g", "ridge"], name
        assert _preds(out) == want, name


def test_an_unknown_extension_needs_the_format_named(tmp_path):
    df = _frame(n=500)
    src, dst = tmp_path / "in.dat", tmp_path / "out.bin"
    df.write_parquet(src)
    with pytest.raises(ValueError, match="cannot tell the format of .*in.dat"):
        po.run(input=src, output=tmp_path / "out.parquet", specs=[_spec()])
    with pytest.raises(
        ValueError, match="input_format 'xml' is not one of parquet, ipc, csv, ndjson"
    ):
        po.run(input=src, input_format="xml", output=dst, specs=[_spec()])
    po.run(input=src, input_format="parquet", output=dst, output_format="csv", specs=[_spec()])
    assert pl.read_csv(dst)["ridge.pred_y"].to_list() == _bank_preds(df)


def test_file_problems_are_the_os_error_for_the_path(tmp_path):
    # Each file `run` touches fails as the OSError subclass for what went
    # wrong, with the path in the message, so a caller can tell "no state yet"
    # from "cannot write here" without parsing text.
    src = tmp_path / "in.parquet"
    _write(src, n=100)
    with pytest.raises(FileNotFoundError, match="nope.parquet"):
        po.run(input=tmp_path / "nope.parquet", output=tmp_path / "o.parquet", specs=[_spec()])
    with pytest.raises(FileNotFoundError, match="writing .*missing"):
        po.run(input=src, output=tmp_path / "missing" / "o.parquet", specs=[_spec()])
    with pytest.raises(FileNotFoundError, match="loading state .*nope.state"):
        po.run(
            input=src,
            output=tmp_path / "o.parquet",
            specs=[_spec()],
            load_state=tmp_path / "nope.state",
        )
    garbage = tmp_path / "garbage.state"
    garbage.write_bytes(b"not a bank")
    with pytest.raises(ValueError, match="loading state .*garbage.state: not a polars-online"):
        po.run(input=src, output=tmp_path / "o.parquet", specs=[_spec()], load_state=garbage)


def test_an_unwritable_save_state_is_refused_before_the_run(tmp_path):
    # Found out at the end, a bad `save_state` would leave the output written
    # and the state lost: the rows would look processed but could not be
    # resumed from. So the directory is checked before a row is read.
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    _write(src, n=100)
    calls = []
    with pytest.raises(FileNotFoundError, match="saving state .*missing.* is not a directory"):
        po.run(
            input=src,
            output=dst,
            specs=[_spec()],
            save_state=tmp_path / "missing" / "bank.state",
            progress=lambda rows, chunks: calls.append(rows),
        )
    assert not dst.exists() and calls == [], "nothing should have run"


def test_an_unknown_config_key_is_refused(tmp_path):
    # A misspelt key in a TOML would otherwise keep its default in silence.
    src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
    _write(src, n=100)
    with pytest.raises(ValueError, match="unknown field `chunk_row`, expected one of"):
        po.run({"chunk_row": 10}, input=src, output=dst, specs=[_spec()])
    with pytest.raises(
        ValueError, match=r"specs\[0\]\.halflfe: unknown field `halflfe`, expected one of `name`"
    ):
        po.run(input=src, output=dst, specs=[{**_spec(), "halflfe": 10.0}])
    with pytest.raises(ValueError, match=r"unknown field `rigde`, expected one of `ridge`"):
        spec = _spec()
        spec["model"] = {**spec["model"], "rigde": 0.1}
        po.run(input=src, output=dst, specs=[spec])
    with pytest.raises(ValueError, match="chunk_rows must be at least 1, got 0"):
        po.run(input=src, output=dst, specs=[_spec()], chunk_rows=0)
