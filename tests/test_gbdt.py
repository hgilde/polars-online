"""``po.gbdt``: XGBoost's and LightGBM's training objects built from a stream
(docs/PLAN.md task 242, docs/GBDT-ADAPTER.md).

Held to each library's own Python route by the hash of the trees it grows:
two routes that hand a library the same values on the same bins grow the
same trees, to the bit, and any difference in a value, a null, a bin edge or
a row's order shows as another hash. The library's own route here is numpy
(pyarrow is unimportable in this session, as for a user of the package;
tests/_site/without_pyarrow.py): XGBoost's ``QuantileDMatrix`` and
LightGBM's ``Dataset`` from the same columns as float arrays, NaN for a null,
an Enum's codes with the categorical feature named. The sample's cuts are
held to their definition, written here from the docstring.
"""

from __future__ import annotations

import gc
import hashlib
import json
import warnings
from collections import Counter

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import xgboost as xgb

import child
from polars_online import _polars_online as native
from polars_online import gbdt

TIER = "mixed"

LEVELS = ["a", "b", "c", "d", "e"]
XGB = {"tree_method": "hist", "max_depth": 4, "eta": 0.3, "max_bin": 256, "nthread": 4, "seed": 0}
LGB = {
    "objective": "regression",
    "num_leaves": 15,
    "learning_rate": 0.3,
    "num_threads": 4,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}
#: Dataset parameters for LightGBM: a sample of 5,000 rows, smaller than the
#: data, so that which rows LightGBM samples shows in the bins.
LGB_DATA = {"verbose": -1, "bin_construct_sample_cnt": 5000}


def frame(n: int = 20_000, seed: int = 0) -> pl.DataFrame:
    """Floats of four shapes (one with 5% nulls, one of few distinct
    values), an integer, a Boolean, an Enum of five uneven levels, a label
    and a weight."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.lognormal(size=n)
    x3 = np.round(rng.normal(size=n), 1)
    i0 = rng.integers(0, 20, size=n).astype(np.int32)
    b0 = rng.random(n) < 0.3
    cat = rng.choice(len(LEVELS), size=n, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    y = (
        2 * x0
        + np.where(x2 > 1.5, 1.0, -0.5)
        + 0.3 * i0
        + cat * 0.7
        + rng.normal(scale=0.5, size=n)
    )
    return pl.DataFrame(
        {
            "x0": x0,
            "x1": pl.Series(x1).set(pl.Series(rng.random(n) < 0.05), None),
            "x2": x2,
            "x3": x3,
            "i0": i0,
            "b0": b0,
            "cat": pl.Series([LEVELS[c] for c in cat], dtype=pl.Enum(LEVELS)),
            "y": y,
            "w": rng.uniform(0.5, 2.0, size=n),
        }
    )


FEATURES = ["x0", "x1", "x2", "x3", "i0", "b0", "cat"]
TYPES = ["float", "float", "float", "float", "int", "i", "c"]


def numpy_x(df: pl.DataFrame, dtype: type = np.float64) -> np.ndarray:
    """The features as the libraries' numpy routes take them."""
    return (
        df.select(
            pl.col(c).to_physical().cast(pl.Float64) if c == "cat" else pl.col(c).cast(pl.Float64)
            for c in FEATURES
        )
        .to_numpy()
        .astype(dtype)
    )


def xgb_trees(bst: xgb.Booster) -> str:
    return hashlib.sha256("\n".join(bst.get_dump(dump_format="json")).encode()).hexdigest()


def lgb_trees(bst: lgb.Booster) -> str:
    dump = json.dumps(bst.dump_model()["tree_info"], sort_keys=True)
    return hashlib.sha256(dump.encode()).hexdigest()


def xgb_own(
    df: pl.DataFrame, ref: xgb.DMatrix | None = None, weight: bool = True
) -> xgb.QuantileDMatrix:
    return xgb.QuantileDMatrix(
        numpy_x(df),
        label=df["y"].to_numpy(),
        weight=df["w"].to_numpy() if weight else None,
        feature_names=FEATURES,
        feature_types=TYPES,
        enable_categorical=True,
        max_bin=256,
        ref=ref,
    )


def lgb_own(df: pl.DataFrame, params: dict, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    return lgb.Dataset(
        numpy_x(df, np.float32),
        label=df["y"].to_numpy(),
        weight=df["w"].to_numpy(),
        feature_name=FEATURES,
        categorical_feature=[FEATURES.index("cat")],
        params={"max_bin": 255, **params},
        reference=reference,
        free_raw_data=False,
    ).construct()


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return frame()


@pytest.fixture(scope="module")
def valid() -> pl.DataFrame:
    return frame(5_000, seed=1)


@pytest.fixture(scope="module")
def files(tmp_path_factory, data) -> dict[str, list[str]]:
    """The data as Arrow IPC and as Parquet, in three files each."""
    d = tmp_path_factory.mktemp("gbdt")
    out: dict[str, list[str]] = {"ipc": [], "parquet": []}
    for k, part in enumerate(data.iter_slices(7_000)):
        p = d / f"part-{k}.arrow"
        part.write_ipc(p)
        out["ipc"].append(str(p))
        q = d / f"part-{k}.parquet"
        part.write_parquet(q, row_group_size=3_000)
        out["parquet"].append(str(q))
    return out


# ----------------------------------------------------------------- XGBoost


def splitmix(i: np.ndarray, seed: int) -> np.ndarray:
    """SplitMix64 of row ``i`` under ``seed``, as the docstring defines the
    sample's hash."""
    with np.errstate(over="ignore"):
        z = i ^ (np.uint64(seed) * np.uint64(0xD1B54A32D192ED03))
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def planted(values: np.ndarray, max_bin: int) -> np.ndarray:
    """The edges the docstring defines from a feature's sample values."""
    s = np.sort(values[np.isfinite(values)])
    distinct = np.unique(s)
    if len(distinct) <= max_bin:
        return distinct
    m = len(s)
    out = [s[0]]
    for j in range(1, max_bin):
        v = s[j * m // max_bin]
        if v > out[-1]:
            out.append(v)
    return np.array(out, dtype=np.float32)


@pytest.mark.parametrize("engine", ["native", "python"])
def test_the_cuts_are_the_defined_sample_s_quantiles(data, engine):
    """XGBoost's cuts are the sample's planted edges, the sample being the
    ``sample_rows`` rows of smallest hash: written from the docstring, not
    from the code."""
    k, max_bin = 3_000, 64
    m = gbdt.xgboost_matrix(
        data, label="y", weight="w", sample_rows=k, max_bin=max_bin, seed=5, engine=engine
    )
    h = splitmix(np.arange(data.height, dtype=np.uint64), 5)
    kept = np.sort(np.argsort(h, kind="stable")[:k])
    x = numpy_x(data, np.float32)[kept]
    indptr, cuts = m.get_quantile_cut()
    for j, name in enumerate(FEATURES):
        got = cuts[indptr[j] : indptr[j + 1]]
        if name == "cat":
            np.testing.assert_array_equal(got, np.arange(len(LEVELS), dtype=np.float32))
            continue
        want = planted(x[:, j], max_bin)
        assert got[0] == -np.inf, name
        np.testing.assert_array_equal(got[1:-1], want[1:], err_msg=name)
    # A feature of few values gets a bin per value it was seen with.
    j = FEATURES.index("i0")
    assert indptr[j + 1] - indptr[j] == len(np.unique(x[:, j])) + 1 == 21


@pytest.mark.parametrize("engine", ["native", "python"])
def test_xgboost_grows_its_own_trees_on_the_same_cuts(data, valid, engine):
    """With the sample's cuts, the trees and the scores of XGBoost's numpy
    route binned on the same cuts; nulls by Arrow validity, the Enum as a
    categorical feature."""
    m = gbdt.xgboost_matrix(data, label="y", weight="w", engine=engine)
    assert m.feature_names == FEATURES and m.feature_types == TYPES
    assert m.num_row() == data.height
    own = xgb_own(data, ref=m)
    a, b = xgb.train(XGB, m, 20), xgb.train(XGB, own, 20)
    assert xgb_trees(a) == xgb_trees(b)
    xv = xgb.DMatrix(
        numpy_x(valid), feature_types=TYPES, feature_names=FEATURES, enable_categorical=True
    )
    assert rmse(a.predict(xv), valid["y"].to_numpy()) == rmse(b.predict(xv), valid["y"].to_numpy())
    split = '"split": "cat", "split_condition": ['
    assert any(split in t for t in a.get_dump(dump_format="json")), "a categorical split"


@pytest.mark.parametrize("engine", ["native", "python"])
def test_without_a_sample_xgboost_sketches_and_grows_its_own_trees(data, engine):
    m = gbdt.xgboost_matrix(data, label="y", weight="w", sample_rows=None, engine=engine)
    own = xgb_own(data)
    i1, c1 = m.get_quantile_cut()
    i2, c2 = own.get_quantile_cut()
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(c1, c2)
    assert xgb_trees(xgb.train(XGB, m, 20)) == xgb_trees(xgb.train(XGB, own, 20))


@pytest.mark.parametrize("engine", ["native", "python"])
def test_a_validation_matrix_is_binned_on_the_training_cuts(data, valid, engine):
    """Evaluation sets, early stopping and a callback through ``xgb.train``,
    with the same record as XGBoost's numpy route on the same cuts."""
    train = gbdt.xgboost_matrix(data, label="y", weight="w", engine=engine)
    vm = gbdt.xgboost_matrix(valid, label="y", weight="w", ref=train, engine=engine)
    assert vm.ref() is train
    np.testing.assert_array_equal(vm.get_quantile_cut()[1], train.get_quantile_cut()[1])
    seen: list[int] = []

    class Count(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            seen.append(epoch)
            return False

    rec_a: dict = {}
    a = xgb.train(
        {**XGB, "eta": 0.8, "max_depth": 6},
        train,
        300,
        evals=[(vm, "valid")],
        early_stopping_rounds=3,
        evals_result=rec_a,
        verbose_eval=False,
        callbacks=[Count()],
    )
    own = xgb_own(data, ref=train)
    own_v = xgb_own(valid, ref=own)
    rec_b: dict = {}
    b = xgb.train(
        {**XGB, "eta": 0.8, "max_depth": 6},
        own,
        300,
        evals=[(own_v, "valid")],
        early_stopping_rounds=3,
        evals_result=rec_b,
        verbose_eval=False,
    )
    assert a.best_iteration == b.best_iteration < 299, "early stopping stopped it"
    assert rec_a == rec_b
    assert seen == list(range(len(rec_a["valid"]["rmse"])))
    assert xgb_trees(a) == xgb_trees(b)


def test_ipc_parquet_a_lazyframe_and_a_frame_grow_the_same_trees(data, files):
    """However the source is given or chunked: IPC and Parquet files, a
    query over them, a frame in memory, and the same frame in many chunks."""
    lazy = pl.scan_parquet(files["parquet"]).with_columns(pl.col("x0") * 1.0)
    chunked = pl.concat(list(data.iter_slices(999)), rechunk=False)
    assert chunked.n_chunks() > 10
    sources = {
        "ipc": files["ipc"],
        "ipc glob": str(files["ipc"][0]).replace("part-0", "part-*"),
        "parquet": files["parquet"],
        "lazy": lazy,
        "frame": data,
        "chunked": chunked,
    }
    hashes = {
        k: xgb_trees(xgb.train(XGB, gbdt.xgboost_matrix(s, label="y", weight="w"), 10))
        for k, s in sources.items()
    }
    assert len(set(hashes.values())) == 1, hashes


def test_the_python_engine_grows_the_native_engine_s_trees(data):
    for sample in (200_000, 2_000, None):
        a = gbdt.xgboost_matrix(data, label="y", sample_rows=sample, engine="native")
        b = gbdt.xgboost_matrix(data, label="y", sample_rows=sample, engine="python")
        assert xgb_trees(xgb.train(XGB, a, 10)) == xgb_trees(xgb.train(XGB, b, 10)), sample


# ---------------------------------------------------------------- LightGBM


@pytest.mark.parametrize("in_memory", [False, True])
@pytest.mark.parametrize("engine", ["native", "python"])
def test_lightgbm_grows_its_own_trees_from_float32(data, valid, engine, in_memory):
    """LightGBM's own sample rows place the same bins, and the same float32
    values grow the same trees, as LightGBM's numpy route from a float32
    array; the Enum as a categorical feature, nulls by validity."""
    ds = gbdt.lightgbm_dataset(
        data, label="y", weight="w", params=LGB_DATA, in_memory=in_memory, engine=engine
    )
    own = lgb_own(data, LGB_DATA)
    assert ds.num_data() == data.height and ds.num_feature() == len(FEATURES)
    assert [ds.feature_num_bin(j) for j in range(len(FEATURES))] == [
        own.feature_num_bin(j) for j in range(len(FEATURES))
    ]
    a, b = lgb.train(LGB, ds, 20), lgb.train(LGB, own, 20)
    assert lgb_trees(a) == lgb_trees(b)
    xv = numpy_x(valid, np.float32)
    assert rmse(a.predict(xv), valid["y"].to_numpy()) == rmse(b.predict(xv), valid["y"].to_numpy())
    assert a.feature_name() == FEATURES
    dumped = json.dumps(a.dump_model()["tree_info"])
    assert '"decision_type": "=="' in dumped, "a categorical split"


@pytest.mark.parametrize("engine", ["native", "python"])
def test_a_validation_dataset_is_binned_on_the_training_bins(data, valid, engine):
    train = gbdt.lightgbm_dataset(data, label="y", weight="w", params=LGB_DATA, engine=engine)
    vd = gbdt.lightgbm_dataset(valid, label="y", weight="w", reference=train, engine=engine)
    assert vd.reference is train
    rec_a: dict = {}
    seen: list[int] = []
    a = lgb.train(
        {**LGB, "learning_rate": 0.9, "num_leaves": 63},
        train,
        300,
        valid_sets=[vd],
        callbacks=[
            lgb.early_stopping(3, verbose=False),
            lgb.record_evaluation(rec_a),
            lambda env: seen.append(env.iteration),
        ],
    )
    own = lgb_own(data, LGB_DATA)
    own_v = lgb_own(valid, LGB_DATA, reference=own)
    rec_b: dict = {}
    b = lgb.train(
        {**LGB, "learning_rate": 0.9, "num_leaves": 63},
        own,
        300,
        valid_sets=[own_v],
        callbacks=[lgb.early_stopping(3, verbose=False), lgb.record_evaluation(rec_b)],
    )
    assert a.best_iteration == b.best_iteration < 299, "early stopping stopped it"
    assert lgb_trees(a) == lgb_trees(b)
    # LightGBM sums the metric in parallel, so its last bit may differ run to run.
    np.testing.assert_allclose(rec_a["valid_0"]["l2"], rec_b["valid_0"]["l2"], rtol=1e-12)
    assert seen == list(range(len(rec_a["valid_0"]["l2"])))
    # A validation set of LightGBM's own on these bins works too.
    mixed = lgb.Dataset(
        numpy_x(valid, np.float32),
        label=valid["y"].to_numpy(),
        reference=train,
        params={"max_bin": 255, **LGB_DATA},
    )
    lgb.train(LGB, train, 3, valid_sets=[mixed])


def test_every_source_and_engine_gives_lightgbm_the_same_trees(data, files):
    lazy = pl.scan_ipc(files["ipc"]).filter(pl.lit(True))
    chunked = pl.concat(list(data.iter_slices(999)), rechunk=False)
    hashes = {}
    for k, s in {
        "ipc": files["ipc"],
        "parquet": files["parquet"],
        "lazy": lazy,
        "chunked": chunked,
    }.items():
        for engine in ("native", "python"):
            for in_memory in (False, True):
                ds = gbdt.lightgbm_dataset(
                    s, label="y", params=LGB_DATA, engine=engine, in_memory=in_memory
                )
                hashes[(k, engine, in_memory)] = lgb_trees(lgb.train(LGB, ds, 10))
    assert len(set(hashes.values())) == 1, hashes


@pytest.mark.parametrize("engine", ["native", "python"])
def test_categorical_names_the_categorical_features(data, valid, engine):
    """An integer column named categorical is one; an Enum left out goes in
    as its codes. A validation set takes its training set's, named or not."""
    ds = gbdt.lightgbm_dataset(data, label="y", categorical=["i0"], params=LGB_DATA, engine=engine)
    assert ds.categorical_feature == [FEATURES.index("i0")]
    own = lgb.Dataset(
        numpy_x(data, np.float32),
        label=data["y"].to_numpy(),
        feature_name=FEATURES,
        categorical_feature=[FEATURES.index("i0")],
        params={"max_bin": 255, **LGB_DATA},
    ).construct()
    assert lgb_trees(lgb.train(LGB, ds, 10)) == lgb_trees(lgb.train(LGB, own, 10))
    vd = gbdt.lightgbm_dataset(valid, label="y", reference=ds, engine=engine)
    assert vd.categorical_feature == ds.categorical_feature
    lgb.train(LGB, ds, 3, valid_sets=[vd])


def test_training_may_not_change_a_dataset_parameter(data):
    """As for any constructed LightGBM dataset whose raw data is gone."""
    ds = gbdt.lightgbm_dataset(data, label="y", params=LGB_DATA)
    with pytest.raises(lgb.basic.LightGBMError, match="max_bin"):
        lgb.train({**LGB, "max_bin": 63}, ds, 2)
    lgb.train({**LGB, "max_bin": 255, "seed": 3}, ds, 2)


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize("engine", ["native", "python"])
@pytest.mark.parametrize("lib", ["xgboost", "lightgbm"])
def test_a_null_or_non_finite_label_or_weight_is_refused_by_name(data, engine, lib):
    def build(df, **kw):
        if lib == "xgboost":
            return gbdt.xgboost_matrix(df, label="y", engine=engine, **kw)
        return gbdt.lightgbm_dataset(df, label="y", engine=engine, **kw)

    bad = data.with_columns(
        pl.when(pl.int_range(pl.len()) == 12_345).then(None).otherwise("y").alias("y")
    )
    with pytest.raises(ValueError, match=r"label column `y` has a null at row 12345"):
        build(bad)
    bad = data.with_columns(
        pl.when(pl.int_range(pl.len()) == 7).then(float("inf")).otherwise("y").alias("y")
    )
    with pytest.raises(ValueError, match=r"label column `y` holds inf at row 7"):
        build(bad)
    bad = data.with_columns(
        pl.when(pl.int_range(pl.len()) == 9).then(-1.0).otherwise("w").alias("w")
    )
    with pytest.raises(ValueError, match=r"weight column `w` holds -1.0 at row 9"):
        build(bad, weight="w")


@pytest.mark.parametrize(
    ("column", "match"),
    [
        (pl.lit("a").alias("s"), r"feature 's' is String"),
        (pl.lit("a").cast(pl.Categorical).alias("s"), r"feature 's' is a pl.Categorical"),
        (pl.lit(0).cast(pl.Datetime).alias("s"), r"feature 's' is Datetime"),
        (pl.lit([1.0]).alias("s"), r"feature 's' is List"),
    ],
)
def test_a_column_the_libraries_do_not_read_is_refused_by_name(data, column, match):
    df = data.with_columns(column)
    for build in (gbdt.xgboost_matrix, gbdt.lightgbm_dataset):
        with pytest.raises(ValueError, match=match):
            build(df, label="y")


def test_the_arguments_are_checked_by_name(data):
    with pytest.raises(ValueError, match="label column 'z' is not in the source"):
        gbdt.xgboost_matrix(data, label="z")
    with pytest.raises(ValueError, match=r"feature columns not in the source: \['q'\]"):
        gbdt.xgboost_matrix(data, label="y", features=["x0", "q"])
    with pytest.raises(ValueError, match="max_bin must be an integer of 2 or more"):
        gbdt.xgboost_matrix(data, label="y", max_bin=1)
    with pytest.raises(ValueError, match="sample_rows must be a positive integer"):
        gbdt.xgboost_matrix(data, label="y", sample_rows=0)
    with pytest.raises(ValueError, match="engine must be one of"):
        gbdt.xgboost_matrix(data, label="y", engine="rust")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ref must be an xgboost.DMatrix"):
        gbdt.xgboost_matrix(data, label="y", ref="train")  # type: ignore[arg-type]
    train = gbdt.xgboost_matrix(data, label="y")
    with pytest.raises(ValueError, match="needs the same features in order"):
        gbdt.xgboost_matrix(data, label="y", features=["x1", "x0"], ref=train)
    with pytest.raises(ValueError, match=r"params names 'max_bin'; pass it as max_bin= instead"):
        gbdt.lightgbm_dataset(data, label="y", params={"max_bin": 63})
    with pytest.raises(ValueError, match="pass it as categorical= instead"):
        gbdt.lightgbm_dataset(data, label="y", params={"categorical_feature": [0]})
    with pytest.raises(ValueError, match="categorical names 'x0', a float column"):
        gbdt.lightgbm_dataset(data, label="y", categorical=["x0"])
    with pytest.raises(ValueError, match="cannot tell the format of 'data.csv'"):
        gbdt.xgboost_matrix("data.csv", label="y")
    for build in (gbdt.xgboost_matrix, gbdt.lightgbm_dataset):
        with pytest.raises(ValueError, match="the source has no rows"):
            build(data.head(0), label="y")


@pytest.mark.parametrize("engine", ["native", "python"])
def test_a_source_whose_passes_differ_is_refused(data, engine):
    """A query that gives other rows each time it runs: the libraries read
    the source more than once, and a pass that differs is refused, not
    binned."""
    calls = Counter()

    def shrink(df: pl.DataFrame) -> pl.DataFrame:
        calls["n"] += 1
        return df.head(df.height - calls["n"])

    for build in (gbdt.xgboost_matrix, gbdt.lightgbm_dataset):
        lazy = data.lazy().map_batches(shrink, streamable=False)
        with pytest.raises(ValueError, match="every pass must give the same rows"):
            build(lazy, label="y", engine=engine)


def test_a_library_or_version_the_native_engine_does_not_call_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="could not be loaded from /no/such/libxgboost"):
        native.GbdtXgboost("/no/such/libxgboost.dylib")
    with pytest.raises(ValueError, match="LightGBM 3.3.5: po.gbdt's native engine calls the C API"):
        native.GbdtLightgbm(gbdt._library_path(lgb.basic._LIB), "3.3.5")
    # A library without the functions: LightGBM's file is no XGBoost.
    with pytest.raises(ValueError, match="has no `XGBoostVersion`"):
        native.GbdtXgboost(gbdt._library_path(lgb.basic._LIB))
    # The floor is 2.1, the first XGBoost whose C API takes Arrow columns.
    assert native.GbdtXgboost(gbdt._library_path(xgb.core._LIB)).version >= (2, 1, 0)


def test_auto_falls_back_to_the_python_engine_with_a_warning(data, monkeypatch):
    def broken(path):
        raise ValueError("XGBoost 9.9.9 at somewhere: not a version")

    monkeypatch.setattr(gbdt, "_xgb_native", broken)
    with pytest.raises(ValueError, match="XGBoost 9.9.9"):
        gbdt.xgboost_matrix(data, label="y", engine="native")
    with pytest.warns(UserWarning, match="cannot load XGBoost natively"):
        m = gbdt.xgboost_matrix(data, label="y", engine="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = gbdt.xgboost_matrix(data, label="y", engine="python")
    assert xgb_trees(xgb.train(XGB, m, 5)) == xgb_trees(xgb.train(XGB, p, 5))


# ------------------------------------------------------------- the handle


def test_each_handle_is_freed_once(data, valid, monkeypatch):
    """By the library's own class, when it is collected; and a build that
    fails frees what it made."""
    freed: Counter = Counter()
    real_x = xgb.core._LIB.XGDMatrixFree
    real_l = lgb.basic._LIB.LGBM_DatasetFree
    monkeypatch.setattr(
        xgb.core._LIB, "XGDMatrixFree", lambda h: freed.update([h.value]) or real_x(h)
    )
    monkeypatch.setattr(
        lgb.basic._LIB, "LGBM_DatasetFree", lambda h: freed.update([h.value]) or real_l(h)
    )
    train = gbdt.xgboost_matrix(data, label="y")
    vm = gbdt.xgboost_matrix(valid, label="y", ref=train)
    xgb.train(XGB, train, 3, evals=[(vm, "v")], verbose_eval=False)
    lt = gbdt.lightgbm_dataset(data, label="y", params=LGB_DATA)
    lv = gbdt.lightgbm_dataset(valid, label="y", reference=lt)
    lgb.train(LGB, lt, 3, valid_sets=[lv])
    handles = [train.handle.value, vm.handle.value, lt._handle.value, lv._handle.value]
    assert not any(freed[h] for h in handles)
    del train, vm, lt, lv
    gc.collect()
    assert [freed[h] for h in handles] == [1, 1, 1, 1]


@pytest.mark.extended(reason="memory: RSS over sixty builds of each library, several seconds")
def test_many_builds_do_not_leak():
    import os

    import psutil

    df = frame(20_000)
    proc = psutil.Process(os.getpid())
    rss = []
    for i in range(60):
        m = gbdt.xgboost_matrix(df, label="y")
        d = gbdt.lightgbm_dataset(df, label="y", params=LGB_DATA)
        del m, d
        gc.collect()
        if i % 10 == 9:
            rss.append(proc.memory_info().rss)
    # A leak of either object grows by its size per build: 20,000 x 7 bins
    # is 140 kB to 560 kB, so 50 builds would be 7 MB or more.
    assert rss[-1] - rss[1] < 4e6, [r / 1e6 for r in rss]


#: A source larger than a build may hold: 6 million rows of 20 float64
#: features and a label, 1.0 GB, as Arrow IPC in record batches of 100,000
#: rows and as Parquet in row groups of as many.
BIG_ROWS, BIG_FEATURES = 6_000_000, 20

#: A child interpreter's build, and the growth of its resident memory at
#: its peak, sampled every 2 ms (psutil, so that Windows reads it too).
_PEAK = """
import threading, time
import psutil
import polars as pl
import polars_online as po
import {lib}
src = {source}
proc = psutil.Process()
peak = [0]
done = [False]
def watch():
    while not done[0]:
        peak[0] = max(peak[0], proc.memory_info().rss)
        time.sleep(0.002)
base = proc.memory_info().rss
threading.Thread(target=watch, daemon=True).start()
out = {build}
done[0] = True
print(peak[0] - base)
"""


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    d = tmp_path_factory.mktemp("gbdt-big")
    rng = np.random.default_rng(3)
    per = BIG_ROWS // 6
    for k in range(6):
        df = pl.DataFrame(
            {f"x{j}": rng.normal(size=per) for j in range(BIG_FEATURES)}
            | {"y": rng.normal(size=per)}
        )
        df.write_ipc(d / f"part-{k}.arrow", record_batch_size=100_000)
        df.write_parquet(d / f"part-{k}.parquet", row_group_size=100_000)
    return d


@pytest.mark.extended(
    reason="memory: a child interpreter builds from a 1 GB source, seconds to write and read"
)
@pytest.mark.parametrize("lib", ["xgboost", "lightgbm"])
@pytest.mark.parametrize(("source", "limit"), [("ipc", 0.75e9), ("parquet", 1.0e9)])
def test_a_source_larger_than_memory_streams(big, lib, source, limit):
    """The build never holds the 1.0 GB source. Arrow IPC files are read a
    record batch at a time, and two batches wait at most: measured, the peak
    grew by 0.46 to 0.48 GB, the binned data (a byte a value, 0.12 GB) with its
    copies and the allocator's retained pages. A Parquet scan reads through
    Polars' streaming engine, whose read-ahead is Polars' own: cut as the
    README's memory section cuts it, measured 0.74 GB. Holding the source
    would be 1.0 GB and the binned data beside it."""
    import subprocess
    import sys

    path = str(big / f"part-*.{'arrow' if source == 'ipc' else 'parquet'}")
    src = repr(path) if source == "ipc" else f"pl.scan_parquet({path!r})"
    build = (
        'po.gbdt.xgboost_matrix(src, label="y")'
        if lib == "xgboost"
        else 'po.gbdt.lightgbm_dataset(src, label="y", params={"verbose": -1}, in_memory=False)'
    )
    env = child.env(POLARS_ROW_GROUP_PREFETCH_SIZE="1", POLARS_MAX_THREADS="4")
    res = subprocess.run(
        [sys.executable, "-c", _PEAK.format(lib=lib, source=src, build=build)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    grew = int(res.stdout.strip().splitlines()[-1])
    assert grew < limit, f"the peak grew by {grew / 1e6:.0f} MB for a 1.0 GB source"
