"""E8: `polars_online.run` — the streaming parquet runner from Python.

Same code path as the `online` CLI (`online_polars::run_config`), so memory is
O(state + chunk) rather than O(data), without spawning a process.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po


def _write(path, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    pl.DataFrame(
        {
            "t": np.arange(float(n)),
            "x0": x0,
            "y": 2 * x0 + 0.1 * rng.standard_normal(n),
            "g": np.where(np.arange(n) % 2 == 0, "a", "b"),
        }
    ).write_parquet(path)


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
    po.run(input=src, output=dst, specs=[_spec()])
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
