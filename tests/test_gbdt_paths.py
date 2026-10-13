"""``po.gbdt`` on the paths no other test took (task 252), each held to the
library's own route by the hash of the trees it grows, as
``tests/test_gbdt.py`` holds the rest:

- every primitive numeric feature type, through both libraries, both of
  LightGBM's routes, and the native engine's own Arrow IPC reader, which
  also takes a label stored as an integer or a Boolean;
- LightGBM's two-pass route (``in_memory=False``) on a reference's bins,
  over record batches that hold none of its sample rows, and over a source
  whose passes differ;
- a refusal, an exception of the source's own and a library's own error,
  each reaching the caller by name;
- the Python engine on an empty source, ``threads`` given to LightGBM, and
  a second IPC file that lacks a column or holds one of another type.
"""

from __future__ import annotations

import inspect
from collections import Counter

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from polars_online import gbdt
from test_gbdt import LGB, LGB_DATA, XGB, frame, lgb_own, lgb_trees, numpy_x, xgb_trees

TIER = "essential"

#: Every primitive numeric type a feature may arrive as, each holding small
#: integers its narrowest type takes, so that the library's own route from a
#: float array reads the same values; each float type twice, with and
#: without nulls, since a column with none hands over no validity.
TYPES = {
    "i8": pl.Int8,
    "i16": pl.Int16,
    "i32": pl.Int32,
    "i64": pl.Int64,
    "u8": pl.UInt8,
    "u16": pl.UInt16,
    "u32": pl.UInt32,
    "u64": pl.UInt64,
    "f32": pl.Float32,
    "f64": pl.Float64,
    "f32_full": pl.Float32,
    "f64_full": pl.Float64,
}


def _write_batches(df: pl.DataFrame, path: object, rows: int) -> None:
    """``df`` as an Arrow IPC file of record batches of ``rows`` rows:
    ``record_batch_size`` where Polars takes it (2.0 on), one batch per
    chunk before."""
    if "record_batch_size" in inspect.signature(pl.DataFrame.write_ipc).parameters:
        df.write_ipc(path, record_batch_size=rows)  # type: ignore[arg-type]
    else:
        pl.concat(list(df.iter_slices(rows)), rechunk=False).write_ipc(path)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def typed() -> pl.DataFrame:
    """A feature of each type, the even ones with 5% nulls (none in the
    ``_full`` ones), and a label that depends on all of them."""
    rng = np.random.default_rng(42)
    n = 12_000
    cols = {}
    y = rng.normal(scale=0.5, size=n)
    for j, (name, dtype) in enumerate(TYPES.items()):
        v = rng.integers(0, 60, size=n)
        y = y + (j % 3 - 1) * 0.05 * v
        s = pl.Series(name, v).cast(dtype)
        if j % 2 == 0 and not name.endswith("_full"):
            s = s.set(pl.Series(rng.random(n) < 0.05), None)
        cols[name] = s
    return pl.DataFrame({**cols, "y": y})


def _xgb_types() -> list[str]:
    return ["float" if n.startswith("f") else "int" for n in TYPES]


def _x(df: pl.DataFrame, dtype: type = np.float64) -> np.ndarray:
    return df.select(pl.col(c).cast(pl.Float64) for c in TYPES).to_numpy().astype(dtype)


def test_every_numeric_feature_type_grows_xgboost_s_own_trees(typed, tmp_path):
    """XGBoost reads each column on its own buffer by its type string, the
    validity bitmap as the mask: the same cuts and trees as its own route
    from a float array, NaN for a null, from a frame and from Arrow IPC."""
    own = xgb.QuantileDMatrix(
        _x(typed),
        label=typed["y"].to_numpy(),
        feature_names=list(TYPES),
        feature_types=_xgb_types(),
        max_bin=256,
    )
    want = xgb_trees(xgb.train(XGB, own, 10))
    path = tmp_path / "typed.arrow"
    _write_batches(typed, path, 5_000)
    for source in (typed, str(path)):
        m = gbdt.xgboost_matrix(source, label="y", sample_rows=None, engine="native")
        np.testing.assert_array_equal(m.get_quantile_cut()[1], own.get_quantile_cut()[1])
        assert xgb_trees(xgb.train(XGB, m, 10)) == want, source
    # The sample's cuts read every type too: the native engine and the
    # Python engine, which share the sampler, grow the same trees.
    a = gbdt.xgboost_matrix(typed, label="y", sample_rows=3_000, engine="native")
    b = gbdt.xgboost_matrix(typed, label="y", sample_rows=3_000, engine="python")
    assert xgb_trees(xgb.train(XGB, a, 10)) == xgb_trees(xgb.train(XGB, b, 10))


@pytest.mark.parametrize("in_memory", [False, True])
def test_every_numeric_feature_type_grows_lightgbm_s_own_trees(typed, tmp_path, in_memory):
    """LightGBM's sample rows and its pushed rows read every type as
    float32, and its Arrow route converts each to float32: the same bins
    and trees as its own route from a float32 array."""
    params = {"verbose": -1, "bin_construct_sample_cnt": 4_000}
    own = lgb.Dataset(
        _x(typed, np.float32),
        label=typed["y"].to_numpy(),
        feature_name=list(TYPES),
        params={"max_bin": 255, **params},
        free_raw_data=False,
    ).construct()
    want = lgb_trees(lgb.train(LGB, own, 10))
    path = tmp_path / "typed.arrow"
    _write_batches(typed, path, 5_000)
    for source in (typed, str(path)):
        ds = gbdt.lightgbm_dataset(
            source, label="y", params=params, in_memory=in_memory, engine="native"
        )
        bins = [ds.feature_num_bin(j) for j in range(len(TYPES))]
        assert bins == [own.feature_num_bin(j) for j in range(len(TYPES))]
        assert lgb_trees(lgb.train(LGB, ds, 10)) == want, source


@pytest.mark.parametrize("label", ["Int32", "UInt8", "Float32", "Boolean"])
def test_an_ipc_label_stored_otherwise_than_float64_is_read_as_float64(tmp_path, label):
    """The native engine reads Arrow IPC files itself, and converts a label
    and a weight that are not Float64 as the plan casts them for any other
    source: the trees are the frame's."""
    df = frame(8_000, seed=4)
    if label == "Boolean":
        df = df.with_columns(y=pl.col("y") > pl.col("y").median(), w=pl.col("w") > 1.0)
    else:
        df = df.with_columns(
            y=pl.col("y").abs().round().cast(getattr(pl, label)),
            w=(pl.col("w") * 2).round().cast(getattr(pl, label)),
        )
    path = tmp_path / "labels.arrow"
    df.write_ipc(path)
    for build, trees, train, params in (
        (gbdt.xgboost_matrix, xgb_trees, lambda d: xgb.train(XGB, d, 5), {}),
        (
            gbdt.lightgbm_dataset,
            lgb_trees,
            lambda d: lgb.train(LGB, d, 5),
            {"params": LGB_DATA},
        ),
    ):
        from_ipc = build(str(path), label="y", weight="w", engine="native", **params)
        from_frame = build(df, label="y", weight="w", engine="native", **params)
        assert trees(train(from_ipc)) == trees(train(from_frame)), (build.__name__, label)


class Broken(Exception):
    """The source's own exception."""


def test_the_source_s_own_exception_reaches_the_caller_as_itself(tmp_path):
    """A query that raises while the native engine reads it: the exception
    the query raised, of its own class, not a message about it. LightGBM
    counts the source first, so its source breaks on a later pass."""
    df = frame(4_000)

    def broken(d: pl.DataFrame) -> pl.DataFrame:
        raise Broken("the source broke")

    with pytest.raises(Broken, match="the source broke"):
        gbdt.xgboost_matrix(df.lazy().map_batches(broken, streamable=False), label="y")
    calls = Counter()

    def later(d: pl.DataFrame) -> pl.DataFrame:
        calls["n"] += 1
        if calls["n"] > 1:
            raise Broken("the source broke on pass 2")
        return d

    lazy = df.lazy().map_batches(later, streamable=False)
    with pytest.raises(Broken, match="the source broke on pass 2"):
        gbdt.lightgbm_dataset(lazy, label="y", params=LGB_DATA, engine="native")


@pytest.mark.parametrize("source", ["frame", "ipc"])
@pytest.mark.parametrize("lib", ["xgboost", "lightgbm"])
def test_a_refusal_part_way_stops_the_reader_and_names_the_row(tmp_path, monkeypatch, source, lib):
    """A null label in an early batch of many: the build stops there with
    the row named, and the reader running ahead of it stops at its next
    batch rather than reading the rest."""
    monkeypatch.setattr(gbdt, "_CHUNK", 500)
    df = frame(20_000).with_columns(
        y=pl.when(pl.int_range(pl.len()) == 1_234).then(None).otherwise("y")
    )
    src: object = df
    if source == "ipc":
        path = tmp_path / "null.arrow"
        _write_batches(df, path, 500)
        src = str(path)
    build = gbdt.xgboost_matrix if lib == "xgboost" else gbdt.lightgbm_dataset
    kw = {"sample_rows": None} if lib == "xgboost" else {"in_memory": False}
    with pytest.raises(ValueError, match="label column `y` has a null at row 1234"):
        build(src, label="y", engine="native", **kw)


@pytest.mark.parametrize("engine", ["native", "python"])
def test_the_two_pass_route_bins_on_a_reference(engine):
    """``in_memory=False`` on a training set's bins: no sample pass, one
    pass of pushes, and the trees and validation scores LightGBM's own
    validation set on the same bins gives."""
    train_df, valid_df = frame(), frame(5_000, seed=1)
    train = gbdt.lightgbm_dataset(train_df, label="y", weight="w", params=LGB_DATA, engine=engine)
    vd = gbdt.lightgbm_dataset(
        valid_df, label="y", weight="w", reference=train, in_memory=False, engine=engine
    )
    own = lgb_own(train_df, LGB_DATA)
    own_v = lgb_own(valid_df, LGB_DATA, reference=own)
    rec_a: dict = {}
    rec_b: dict = {}
    a = lgb.train(LGB, train, 15, valid_sets=[vd], callbacks=[lgb.record_evaluation(rec_a)])
    b = lgb.train(LGB, own, 15, valid_sets=[own_v], callbacks=[lgb.record_evaluation(rec_b)])
    assert lgb_trees(a) == lgb_trees(b)
    np.testing.assert_allclose(rec_a["valid_0"]["l2"], rec_b["valid_0"]["l2"], rtol=1e-12)


def test_record_batches_that_hold_no_sample_row_are_passed_over(tmp_path):
    """LightGBM samples 50 of 20,000 rows, so most of 100 record batches of
    200 rows hold none of them: the sample pass gathers the 50 wherever they
    fall, and the bins are LightGBM's own."""
    df = frame()
    path = tmp_path / "batches.arrow"
    _write_batches(df, path, 200)
    params = {"verbose": -1, "bin_construct_sample_cnt": 50}
    ds = gbdt.lightgbm_dataset(str(path), label="y", weight="w", params=params, in_memory=False)
    own = lgb_own(df, params)
    assert [ds.feature_num_bin(j) for j in range(7)] == [own.feature_num_bin(j) for j in range(7)]
    assert lgb_trees(lgb.train(LGB, ds, 10)) == lgb_trees(lgb.train(LGB, own, 10))


def _moving_cases() -> list[tuple[str, str, bool, int]]:
    """``(engine, change, on_reference, start)``: the source's rows change
    from its ``start``-th run on. It runs once to be counted, then once per
    pass: the native engine samples and pushes, or on a reference's bins
    only pushes; the Python engine gathers the sample and the labels, then
    LightGBM reads its ``Sequence``. Every pass after the count is a case."""
    out = []
    for engine in ("native", "python"):
        for on_reference in (False, True):
            runs = 2 if engine == "native" and on_reference else 3
            for start in range(2, runs + 1):
                for change in ("shrink", "grow"):
                    out.append((engine, change, on_reference, start))
    return out


@pytest.mark.parametrize(("engine", "change", "on_reference", "start"), _moving_cases())
def test_the_two_pass_route_refuses_a_source_whose_passes_differ(
    engine, change, on_reference, start
):
    """A query that gives other rows each time it runs: counted at one
    number, any later pass that gives fewer or more is refused, whether it
    is the sample pass or the pass of pushes."""
    data = frame(6_000)
    reference = (
        gbdt.lightgbm_dataset(data, label="y", params=LGB_DATA, engine=engine)
        if on_reference
        else None
    )
    calls = Counter()

    def moving(df: pl.DataFrame) -> pl.DataFrame:
        calls["n"] += 1
        if calls["n"] < start:
            return df
        return df.head(df.height - 1) if change == "shrink" else pl.concat([df, df.head(1)])

    lazy = data.lazy().map_batches(moving, streamable=False)
    with pytest.raises(ValueError, match="every pass must give the same rows"):
        gbdt.lightgbm_dataset(lazy, label="y", reference=reference, in_memory=False, engine=engine)
    assert calls["n"] >= start, "the change was reached"


def test_a_library_s_own_error_is_a_value_error_with_its_first_line():
    """What a library refuses reaches the caller as ``ValueError``, saying
    which library and what it was doing, with the library's first line."""
    data = frame(3_000)
    for in_memory in (False, True):
        with pytest.raises(
            ValueError, match=r"LightGBM, .*Check failed: \(min_data_in_bin\) > \(0\)"
        ):
            gbdt.lightgbm_dataset(
                data,
                label="y",
                params={"verbose": -1, "min_data_in_bin": 0},
                in_memory=in_memory,
                engine="native",
            )
    # A reference matrix of XGBoost's own records no feature names, so only
    # XGBoost can tell that it has another number of features.
    ref = xgb.QuantileDMatrix(np.random.default_rng(0).random((100, 3)), label=np.zeros(100))
    assert ref.feature_names is None
    with pytest.raises(
        ValueError, match="XGBoost, building the QuantileDMatrix: .*different number of features"
    ):
        gbdt.xgboost_matrix(data, label="y", ref=ref, engine="native")


def test_the_python_engine_refuses_an_empty_source():
    empty = frame(10).head(0)
    for sample_rows in (200_000, None):
        with pytest.raises(ValueError, match="the source has no rows"):
            gbdt.xgboost_matrix(empty, label="y", sample_rows=sample_rows, engine="python")


@pytest.mark.parametrize("engine", ["native", "python"])
def test_threads_given_to_lightgbm_bins_the_same(engine):
    """``threads`` is LightGBM's ``num_threads`` while it bins; the bins,
    and so the trees, do not depend on it."""
    data = frame(8_000)
    one = gbdt.lightgbm_dataset(
        data, label="y", weight="w", params=LGB_DATA, threads=1, engine=engine
    )
    own = lgb_own(data, LGB_DATA)
    assert lgb_trees(lgb.train(LGB, one, 10)) == lgb_trees(lgb.train(LGB, own, 10))
    assert numpy_x(data).shape[1] == one.num_feature()


def test_a_second_ipc_file_that_differs_is_refused_by_name(tmp_path):
    """The native engine reads each Arrow IPC file by the columns it names:
    a later file without one, or with one of a type the libraries do not
    read, is refused naming the file and the column."""
    df = frame(4_000)
    first = tmp_path / "part-0.arrow"
    df.write_ipc(first)
    missing = tmp_path / "part-1.arrow"
    df.drop("x1").write_ipc(missing)
    with pytest.raises(ValueError, match=r"part-1.arrow has no column `x1`"):
        gbdt.xgboost_matrix([str(first), str(missing)], label="y", engine="native")
    stringy = tmp_path / "part-2.arrow"
    df.with_columns(pl.col("x1").cast(pl.String)).write_ipc(stringy)
    with pytest.raises(
        ValueError,
        match=r"column `x1` is Utf8View in the Arrow IPC file, which is not a column the "
        r"libraries read",
    ):
        gbdt.xgboost_matrix([str(first), str(stringy)], label="y", engine="native")
