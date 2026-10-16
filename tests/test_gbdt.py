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
from fractions import Fraction

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
    """Floats of four shapes (one of few distinct values), an integer, a
    Boolean, an Enum of five uneven levels, a label and a weight. The label
    depends on ``x0``, ``x2``, ``i0``, ``b0`` and ``cat``, so the trees
    split on each of them, and ``x0`` has 5% nulls: a null read as 0 there
    moves a split, where one on a feature the trees never split on would
    leave the tree hash unchanged (review 7, C-2). ``x1`` and ``x3`` are
    noise."""
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
        + 1.5 * b0
        + cat * 0.7
        + rng.normal(scale=0.5, size=n)
    )
    return pl.DataFrame(
        {
            "x0": pl.Series(x0).set(pl.Series(rng.random(n) < 0.05), None),
            "x1": x1,
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
    df: pl.DataFrame,
    ref: xgb.DMatrix | None = None,
    weight: bool = True,
    x: np.ndarray | None = None,
) -> xgb.QuantileDMatrix:
    return xgb.QuantileDMatrix(
        numpy_x(df) if x is None else x,
        label=df["y"].to_numpy(),
        weight=df["w"].to_numpy() if weight else None,
        feature_names=FEATURES,
        feature_types=TYPES,
        enable_categorical=True,
        max_bin=256,
        ref=ref,
    )


def lgb_own(
    df: pl.DataFrame,
    params: dict,
    reference: lgb.Dataset | None = None,
    x: np.ndarray | None = None,
) -> lgb.Dataset:
    return lgb.Dataset(
        numpy_x(df, np.float32) if x is None else x,
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


def planted(values: np.ndarray, max_bin: int, weights: np.ndarray | None = None) -> np.ndarray:
    """The edges the docstring defines from a feature's sample values and,
    with ``weights``, the sample's estimated weights: the lowest value of
    positive weight, then for ``j = 1 .. max_bin - 1`` the first sorted
    value whose cumulative weight exceeds ``j / max_bin`` of the total, each
    distinct; or every distinct value when there are at most ``max_bin``.
    Exact arithmetic (``Fraction``), so that nothing here shares the
    implementation's rounding."""
    w = np.ones(len(values)) if weights is None else weights
    keep = np.isfinite(values) & (w > 0)
    v, w = values[keep], w[keep]
    if len(v) == 0:
        return np.array([0.0], dtype=np.float32)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    distinct = np.unique(v)
    if len(distinct) <= max_bin:
        return distinct
    total = sum(Fraction(float(x)) for x in w)
    out = [v[0]]
    acc, i = Fraction(float(w[0])), 0
    for j in range(1, max_bin):
        target = total * j / max_bin
        while acc <= target:
            i += 1
            acc += Fraction(float(w[i]))
        if v[i] > out[-1]:
            out.append(v[i])
    return np.array(out, dtype=np.float32)


def priority_sample(w: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The docstring's sample under row weights, from its definition: row
    ``i`` has priority ``w_i / u_i`` with ``u_i = (2 h(i) + 1) / 2**65``,
    the ``k`` rows of largest priority are kept (an earlier row before a
    later one of equal priority), and a kept row stands for ``max(w_i,
    tau)``, ``tau`` the largest priority not kept (0 when every row is
    kept). Returns the kept rows, in order, and their weights, exact."""
    h = splitmix(np.arange(len(w), dtype=np.uint64), seed)
    prio = [Fraction(float(wi)) * 2**65 / (2 * int(hi) + 1) for wi, hi in zip(w, h, strict=True)]
    order = sorted(range(len(w)), key=lambda i: (-prio[i], i))
    kept = np.sort(np.array(order[:k]))
    tau = prio[order[k]] if len(w) > k else Fraction(0)
    est = np.array([max(Fraction(float(w[i])), tau) for i in kept], dtype=object)
    return kept, est


def planted_exact(values: np.ndarray, est: np.ndarray, max_bin: int) -> np.ndarray:
    """``planted`` over exact weights (``Fraction``s)."""
    keep = np.isfinite(values) & np.array([e > 0 for e in est], dtype=bool)
    v, w = values[keep], est[keep]
    if len(v) == 0:
        return np.array([0.0], dtype=np.float32)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    distinct = np.unique(v)
    if len(distinct) <= max_bin:
        return distinct
    total = sum(w)
    out = [v[0]]
    acc, i = w[0], 0
    for j in range(1, max_bin):
        target = total * j / max_bin
        while acc <= target:
            i += 1
            acc += w[i]
        if v[i] > out[-1]:
            out.append(v[i])
    return np.array(out, dtype=np.float32)


def assert_cuts(m: xgb.DMatrix, x: np.ndarray, want_of) -> None:
    indptr, cuts = m.get_quantile_cut()
    for j, name in enumerate(FEATURES):
        got = cuts[indptr[j] : indptr[j + 1]]
        if name == "cat":
            np.testing.assert_array_equal(got, np.arange(len(LEVELS), dtype=np.float32))
            continue
        want = want_of(x[:, j])
        assert got[0] == -np.inf, name
        np.testing.assert_array_equal(got[1:-1], want[1:], err_msg=name)


@pytest.mark.parametrize("engine", ["native", "python"])
def test_the_cuts_are_the_defined_sample_s_quantiles(data, engine):
    """Without weights, XGBoost's cuts are the sample's planted edges, the
    sample being the ``sample_rows`` rows of smallest hash: written from
    the docstring, not from the code."""
    k, max_bin = 3_000, 64
    m = gbdt.xgboost_matrix(data, label="y", sample_rows=k, max_bin=max_bin, seed=5, engine=engine)
    h = splitmix(np.arange(data.height, dtype=np.uint64), 5)
    kept = np.sort(np.argsort(h, kind="stable")[:k])
    x = numpy_x(data, np.float32)[kept]
    assert_cuts(m, x, lambda col: planted(col, max_bin))
    # A feature of few values gets a bin per value it was seen with.
    j = FEATURES.index("i0")
    indptr = m.get_quantile_cut()[0]
    assert indptr[j + 1] - indptr[j] == len(np.unique(x[:, j])) + 1 == 21


@pytest.mark.parametrize("engine", ["native", "python"])
@pytest.mark.parametrize("weights", ["uniform", "skewed"])
def test_weighted_cuts_are_the_defined_priority_sample_s_quantiles(data, engine, weights):
    """With ``weight=``, the cuts are the weighted quantiles of a priority
    sample (review 7, C-4: the sample once ignored the weights, where
    XGBoost's own sketch is weighted). Written from the docstring in exact
    arithmetic. ``uniform`` is the fixture's weights, 0.5 to 2, every one
    below ``tau``, so each kept row stands for ``tau``; ``skewed`` gives the
    rows of ``x2 > 3`` a weight of 400, which puts them all in the sample at
    their own weight."""
    k, max_bin = 3_000, 64
    df = data
    if weights == "skewed":
        df = data.with_columns(w=pl.when(pl.col("x2") > 3).then(400.0).otherwise(pl.col("w")))
    m = gbdt.xgboost_matrix(
        df, label="y", weight="w", sample_rows=k, max_bin=max_bin, seed=5, engine=engine
    )
    w = df["w"].to_numpy()
    kept, est = priority_sample(w, k, 5)
    if weights == "skewed":
        heavy = np.flatnonzero(w == 400.0)
        assert len(heavy) > 100 and np.isin(heavy, kept).all(), "a row above tau is always kept"
        assert len({e for e in est}) > 1, "the kept rows stand for different weights"
    x = numpy_x(df, np.float32)[kept]
    assert_cuts(m, x, lambda col: planted_exact(col, est, max_bin))


def test_weighted_cuts_are_numpy_s_weighted_quantiles():
    """A third-party oracle: with every row in the sample, each cut is
    numpy's weighted quantile (``method="inverted_cdf"``, the first value
    whose cumulative weight reaches the level) at ``j / max_bin``. On
    weights tied to the value (review 7, C-4: 50 above 1.0 put 14 of
    XGBoost's 15 cuts above 1.0 and 2 of the sample's), on rows of weight 0
    far from the rest (hard rule 9: they place no cut), and on one row
    carrying nearly all the weight."""
    rng = np.random.default_rng(11)
    n, max_bin = 20_000, 16
    x = rng.normal(size=n)
    cases = {
        "weight 50 above 1.0": (x, np.where(x > 1.0, 50.0, 1.0)),
        "weight exp(2x)": (x, np.exp(2 * x)),
    }
    far = np.where(rng.random(n) < 0.9, rng.normal(10, 1, n), x)
    cases["weight 0 on 90% of rows, far away"] = (
        far,
        np.where(far > 5, 0.0, rng.uniform(0.5, 2, n)),
    )
    heavy = rng.uniform(0.5, 2, n)
    heavy[123] = 1e9
    cases["one row of weight 1e9"] = (x, heavy)
    levels = np.arange(1, max_bin) / max_bin
    for name, (v, w) in cases.items():
        df = pl.DataFrame({"x": v, "y": v, "w": w})
        m = gbdt.xgboost_matrix(df, label="y", weight="w", max_bin=max_bin)
        indptr, cuts = m.get_quantile_cut()
        got = cuts[indptr[0] + 1 : indptr[1] - 1]
        x32 = v.astype(np.float32)
        q = np.quantile(x32, levels, weights=w, method="inverted_cdf")
        want = np.unique(q[q > x32[w > 0].min()])
        np.testing.assert_array_equal(got, want, err_msg=name)
        if name == "weight 50 above 1.0":
            assert (got > 1.0).sum() == 14, got
        if name == "one row of weight 1e9":
            np.testing.assert_array_equal(got, [x32[123]])


@pytest.mark.parametrize("engine", ["native", "python"])
def test_xgboost_grows_its_own_trees_on_the_same_cuts(data, valid, engine):
    """The cuts against XGBoost's own weighted sketch of the same rows and
    weights, and the trees and scores against XGBoost's numpy route binned
    on the sample's cuts; nulls by Arrow validity, the Enum as a categorical
    feature. The sketch is approximate and the sample's quantiles exact
    here (every row is in the sample), so the continuous features' cuts are
    held to within 1/512 of the weight of XGBoost's own, where a bin holds
    1/256 of it, and the trees, which see any moved cut, to the numpy route
    on the same cuts. The cuts once ignored the weights (review 7, C-4):
    with the rows of ``x2 > 3`` at weight 400, ``x2``'s cuts sat up to 0.84
    of the weight from XGBoost's own, and at the fixture's own weights 0.0027
    (0.0006 now)."""
    skewed = data.with_columns(w=pl.when(pl.col("x2") > 3).then(400.0).otherwise(pl.col("w")))
    for df in (data, skewed):
        m = gbdt.xgboost_matrix(df, label="y", weight="w", engine=engine)
        assert m.feature_names == FEATURES and m.feature_types == TYPES
        assert m.num_row() == df.height
        i1, c1 = m.get_quantile_cut()
        i2, c2 = xgb_own(df).get_quantile_cut()
        np.testing.assert_array_equal(i1, i2)
        x, w = numpy_x(df, np.float32), df["w"].to_numpy()
        for j, name in enumerate(FEATURES):
            a, b = c1[i1[j] : i1[j + 1]], c2[i2[j] : i2[j + 1]]
            if name in ("x3", "i0", "b0", "cat"):
                np.testing.assert_array_equal(
                    a, b, err_msg=f"{name}: a bin per value, as XGBoost's"
                )
                continue
            ok = np.isfinite(x[:, j])
            xs, ws = x[ok, j], w[ok]

            def below(cuts, xs=xs, ws=ws):
                return np.array([ws[xs < c].sum() for c in cuts[1:-1]]) / ws.sum()

            gap = np.abs(below(a) - below(b)).max()
            assert gap < 1 / 512, f"{name}: a cut {gap:.5f} of the weight from XGBoost's own"
        own = xgb_own(df, ref=m)
        a_, b_ = xgb.train(XGB, m, 20), xgb.train(XGB, own, 20)
        assert xgb_trees(a_) == xgb_trees(b_)
        xv = xgb.DMatrix(
            numpy_x(valid), feature_types=TYPES, feature_names=FEATURES, enable_categorical=True
        )
        y = valid["y"].to_numpy()
        assert rmse(a_.predict(xv), y) == rmse(b_.predict(xv), y)
        split = '"split": "cat", "split_condition": ['
        assert any(split in t for t in a_.get_dump(dump_format="json")), "a categorical split"


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
        for weight in (None, "w"):
            kw = dict(label="y", weight=weight, sample_rows=sample)
            a = gbdt.xgboost_matrix(data, **kw, engine="native")
            b = gbdt.xgboost_matrix(data, **kw, engine="python")
            assert xgb_trees(xgb.train(XGB, a, 10)) == xgb_trees(xgb.train(XGB, b, 10)), kw


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


# -------------------------------------------------------------- the oracle


@pytest.mark.parametrize("lib", ["xgboost", "lightgbm"])
def test_the_tree_hash_sees_a_null_read_as_zero(data, lib):
    """A planted defect: the library's own route with every null of ``x0``
    read as 0 grows other trees than po.gbdt's, so the hash the tests above
    compare would catch a null handed over as 0; and the trees split on
    ``x0`` and on the Boolean, which the hash cannot see otherwise (review 7,
    C-2: with the nulls on a feature the trees never split on, LightGBM's
    hash was the same with them read as 0)."""
    j = FEATURES.index("x0")
    if lib == "xgboost":
        m = gbdt.xgboost_matrix(data, label="y", weight="w")
        bst = xgb.train(XGB, m, 20)
        ours = xgb_trees(bst)
        splits = bst.get_score(importance_type="weight")
        x = numpy_x(data)
        assert np.isnan(x[:, j]).sum() > 500
        x[np.isnan(x[:, j]), j] = 0.0
        theirs = xgb_trees(xgb.train(XGB, xgb_own(data, ref=m, x=x), 20))
    else:
        ds = gbdt.lightgbm_dataset(data, label="y", weight="w", params=LGB_DATA)
        bst = lgb.train(LGB, ds, 20)
        ours = lgb_trees(bst)
        splits = dict(zip(FEATURES, bst.feature_importance("split"), strict=True))
        x = numpy_x(data, np.float32)
        assert np.isnan(x[:, j]).sum() > 500
        x[np.isnan(x[:, j]), j] = 0.0
        theirs = lgb_trees(lgb.train(LGB, lgb_own(data, LGB_DATA, x=x), 20))
    assert splits["x0"] > 0 and splits["b0"] > 0, splits
    assert ours != theirs, "a null read as 0 would go unseen"


@pytest.mark.parametrize("engine", ["native", "python"])
def test_a_ref_or_reference_on_an_enum_of_other_categories_is_refused_by_name(data, valid, engine):
    """A ``pl.Enum``'s codes mean its categories in order, and a validation
    set is binned on the training cuts by its codes: the same Enum is
    accepted, one listing the levels in another order is refused by name,
    and so is a reference with the features in another order (review 7,
    C-1: a reversed Enum was accepted, and predicted the levels' effects
    permuted). A reference built by the library itself is checked by its
    feature names alone."""
    reordered = valid.with_columns(pl.col("cat").cast(pl.String).cast(pl.Enum(LEVELS[::-1])))
    enum = (
        r"as a pl.Enum of \['a', 'b', 'c', 'd', 'e'\].*as a pl.Enum of \['e', 'd', 'c', 'b', 'a'\]"
    )
    train = gbdt.xgboost_matrix(data, label="y", weight="w", engine=engine)
    gbdt.xgboost_matrix(valid, label="y", weight="w", ref=train, engine=engine)
    with pytest.raises(ValueError, match=r"ref was built with 'cat' " + enum):
        gbdt.xgboost_matrix(reordered, label="y", weight="w", ref=train, engine=engine)
    as_codes = valid.with_columns(pl.col("cat").to_physical())
    with pytest.raises(ValueError, match=r"'cat' as a pl.Enum of .* has it as not a pl.Enum"):
        gbdt.xgboost_matrix(as_codes, label="y", weight="w", ref=train, engine=engine)
    own = xgb_own(data)
    gbdt.xgboost_matrix(reordered, label="y", weight="w", ref=own, engine=engine)
    with pytest.raises(ValueError, match="needs the same features in order"):
        gbdt.xgboost_matrix(
            valid, label="y", weight="w", features=FEATURES[::-1], ref=own, engine=engine
        )

    td = gbdt.lightgbm_dataset(data, label="y", weight="w", params=LGB_DATA, engine=engine)
    gbdt.lightgbm_dataset(valid, label="y", weight="w", reference=td, engine=engine)
    with pytest.raises(ValueError, match=r"reference was built with 'cat' " + enum):
        gbdt.lightgbm_dataset(reordered, label="y", weight="w", reference=td, engine=engine)
    with pytest.raises(
        ValueError, match="a dataset binned on another's bins needs the same features"
    ):
        gbdt.lightgbm_dataset(
            valid, label="y", weight="w", features=FEATURES[::-1], reference=td, engine=engine
        )
    own_d = lgb_own(data, LGB_DATA)
    gbdt.lightgbm_dataset(reordered, label="y", weight="w", reference=own_d, engine=engine)
    with pytest.raises(ValueError, match="needs the same features in order"):
        gbdt.lightgbm_dataset(
            valid, label="y", weight="w", features=FEATURES[::-1], reference=own_d, engine=engine
        )


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


_BOTH = ("xgboost", "lightgbm")
_ENGINES = r"\('auto', 'native', 'python'\)"


@pytest.mark.parametrize(
    ("libs", "source", "kw", "exc", "match"),
    [
        (
            _BOTH,
            {"x0": [1.0]},
            {},
            TypeError,
            r"source must be paths, a LazyFrame or a DataFrame, not dict",
        ),
        (_BOTH, [], {}, ValueError, r"source names no files"),
        (
            _BOTH,
            ["a.parquet", "b.arrow"],
            {},
            ValueError,
            r"source mixes Arrow IPC and Parquet files; pass one kind",
        ),
        (
            _BOTH,
            "/nonexistent/*.arrow",
            {},
            FileNotFoundError,
            r"no file matches '/nonexistent/\*.arrow'",
        ),
        (
            _BOTH,
            "data",
            dict(features="x0"),
            TypeError,
            r"features is a list of column names, not one name",
        ),
        (
            _BOTH,
            "data",
            dict(features=["x0", "x1", "x0"]),
            ValueError,
            r"a feature is named twice in \['x0', 'x1', 'x0'\]",
        ),
        (
            _BOTH,
            "data",
            dict(features=["x0", "y"]),
            ValueError,
            r"\['y'\] named as a feature and as the label or weight",
        ),
        (
            _BOTH,
            "data",
            dict(features=["w", "x0"], weight="w"),
            ValueError,
            r"\['w'\] named as a feature and as the label or weight",
        ),
        (
            _BOTH,
            "label only",
            {},
            ValueError,
            r"the source has no feature columns besides the label",
        ),
        (
            _BOTH,
            "with a string",
            dict(label="s"),
            ValueError,
            r"the label column 's' is String, not numeric",
        ),
        (
            _BOTH,
            "with a string",
            dict(weight="s"),
            ValueError,
            r"the weight column 's' is String, not numeric",
        ),
        (
            _BOTH,
            "data",
            dict(weight="zz"),
            ValueError,
            r"the weight column 'zz' is not in the source",
        ),
        (
            _BOTH,
            "data",
            dict(threads=0),
            ValueError,
            r"threads must be a positive integer or None, not 0",
        ),
        (
            _BOTH,
            "data",
            dict(threads=True),
            ValueError,
            r"threads must be a positive integer or None, not True",
        ),
        (
            _BOTH,
            "data",
            dict(threads="4"),
            ValueError,
            r"threads must be a positive integer or None, not '4'",
        ),
        (
            _BOTH,
            "data",
            dict(max_bin="8"),
            ValueError,
            r"max_bin must be an integer of 2 or more, not '8'",
        ),
        (
            _BOTH,
            "data",
            dict(max_bin=True),
            ValueError,
            r"max_bin must be an integer of 2 or more, not True",
        ),
        (
            _BOTH,
            "data",
            dict(engine="rust"),
            ValueError,
            r"engine must be one of " + _ENGINES + r", not 'rust'",
        ),
        (
            ("xgboost",),
            "data",
            dict(sample_rows=True),
            ValueError,
            r"sample_rows must be a positive integer or None, not True",
        ),
        (
            ("xgboost",),
            "data",
            dict(sample_rows=2.5),
            ValueError,
            r"sample_rows must be a positive integer or None, not 2.5",
        ),
        (
            ("lightgbm",),
            "data",
            dict(categorical="i0"),
            TypeError,
            r"categorical is a list of column names, not one name",
        ),
        (
            ("lightgbm",),
            "data",
            dict(categorical=["zz"]),
            ValueError,
            r"categorical names 'zz', which is not a feature",
        ),
        (
            ("lightgbm",),
            "data",
            dict(in_memory=1),
            ValueError,
            r"in_memory must be True, False or 'auto', not 1",
        ),
        (
            ("lightgbm",),
            "data",
            dict(in_memory="yes"),
            ValueError,
            r"in_memory must be True, False or 'auto', not 'yes'",
        ),
        (
            ("lightgbm",),
            "data",
            dict(reference="train"),
            TypeError,
            r"reference must be a lightgbm.Dataset, not str",
        ),
        (
            ("lightgbm",),
            "data",
            dict(params={"num_threads": 2}),
            ValueError,
            r"params names 'num_threads'; pass it as threads= instead",
        ),
    ],
)
def test_it_refuses(data, libs, source, kw, exc, match):
    """Every refusal of an argument, by name, through the public functions
    (review 7, E-3: fourteen of them had no test). ``in_memory=1`` is
    refused by type: ``1 in (True, False)`` holds in Python, and 1 is not an
    answer."""
    frames = {
        "data": data,
        "label only": data.select("y"),
        "with a string": data.with_columns(pl.lit("a").alias("s")),
    }
    src = frames.get(source, source) if isinstance(source, str) else source
    for lib in libs:
        build = gbdt.xgboost_matrix if lib == "xgboost" else gbdt.lightgbm_dataset
        with pytest.raises(exc, match=match):
            build(src, **{"label": "y", **kw})


def test_a_lightgbm_whose_dataset_holds_its_handle_elsewhere_is_refused_by_name(data, monkeypatch):
    """The native engine hands its dataset to ``lightgbm.Dataset`` through
    the private ``_handle``; a LightGBM whose class keeps it elsewhere is
    refused by name, with the Python engine offered."""

    class Elsewhere(lgb.Dataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._native = self.__dict__.pop("_handle")

    monkeypatch.setattr(lgb, "Dataset", Elsewhere)
    with pytest.raises(
        ValueError, match=r"holds its native handle elsewhere than `_handle` \(lightgbm"
    ):
        gbdt.lightgbm_dataset(data, label="y", params=LGB_DATA, engine="native")


def test_a_package_that_does_not_say_which_file_it_loaded_is_refused():
    """The native engine opens the library by the path the Python package
    loaded it from (``xgboost.core._LIB.path``, ``lightgbm.basic._LIB._name``);
    a package that records neither is refused, and ``engine="auto"`` falls
    back to the Python engine on that refusal."""
    with pytest.raises(ValueError, match="does not say which file it loaded"):
        gbdt._library_path(object())
    assert gbdt._library_path(xgb.core._LIB) == str(xgb.core._LIB.path)


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


@pytest.mark.extended(reason="memory: RSS over hundreds of builds of each library, seconds")
@pytest.mark.serial
def test_many_builds_do_not_leak():
    """A leak of either object grows by its size per build: 20,000 rows of
    7 bins is 140 kB to 560 kB, 2 to 8 times the limit of 70 kB a build
    (the fixed window this used allowed 100 kB). The check is
    `test_ffi_memory`'s, which measures only once the allocator's ramp is
    over and finds where that is. In a process that has run little before
    this test, as when the gate runs the `serial` tests on their own, RSS
    rose for about 100 builds before it held (2026-10-11), and the window,
    builds 20 to 60, read that ramp as a leak. Blocks of 25 builds: in
    blocks of 10, RSS swung by a megabyte from one block to the next with
    no leak, a 100 kB a build either way."""
    from test_ffi_memory import assert_plateaus

    df = frame(20_000)

    def build():
        m = gbdt.xgboost_matrix(df, label="y")
        d = gbdt.lightgbm_dataset(df, label="y", params=LGB_DATA)
        del m, d

    assert_plateaus(build, per_block=25, kb_per_iter=70.0)


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
        # One record batch per chunk, 100,000 rows each: `write_ipc` has no
        # `record_batch_size` before py-polars 2.0, and the floor is 1.37.0.
        # Measured 2026-10-12 on 1.34.0 and 2.0.0: 450,000 rows in chunks of
        # 100,000 wrote batches of 100,000 x 4 and 50,000 on both.
        batches = [df.slice(i, 100_000) for i in range(0, df.height, 100_000)]
        pl.concat(batches, rechunk=False).write_ipc(d / f"part-{k}.arrow")
        df.write_parquet(d / f"part-{k}.parquet", row_group_size=100_000)
    return d


def _build_peak(big, lib: str, source: str, parts: str = "*") -> int:
    """The peak resident growth, in bytes, of one ``po.gbdt`` build in a
    child interpreter, from the module's files ``part-<parts>``."""
    import subprocess
    import sys

    path = str(big / f"part-{parts}.{'arrow' if source == 'ipc' else 'parquet'}")
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
        encoding="utf-8",
        env=env,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return int(res.stdout.strip().splitlines()[-1])


# One process, so the 2 GB the module's `big` fixture writes is written once,
# and alone, since each case holds most of a gigabyte at its peak.
@pytest.mark.extended(
    reason="memory: a child interpreter builds from a 1 GB source, seconds to write and read"
)
@pytest.mark.serial
@pytest.mark.parametrize("lib", ["xgboost", "lightgbm"])
def test_an_arrow_source_larger_than_memory_streams(big, lib):
    """The build never holds the 1.0 GB source. Arrow IPC files are read by
    this library, a record batch at a time, and two batches wait at most:
    measured, the peak grew by 0.46 to 0.48 GB, the binned data (a byte a
    value, 0.12 GB) with its copies and the allocator's retained pages."""
    grew = _build_peak(big, lib, "ipc")
    assert grew < 0.75e9, f"the peak grew by {grew / 1e6:.0f} MB for a 1.0 GB source"


@pytest.mark.extended(reason="memory: two child interpreters build from 0.5 and 1 GB sources")
@pytest.mark.serial
@pytest.mark.parametrize("lib", ["xgboost", "lightgbm"])
def test_a_parquet_source_larger_than_memory_streams(big, lib):
    """The build never holds a Parquet source: its peak grows with the
    source by what the library keeps, not by the source. A scan reads
    through Polars' streaming engine, whose read-ahead is Polars' own and
    the same at any size, but not the same in every Polars or on every
    machine: one 1 GB build peaked at 0.76 GB on py-polars 2.0.0 and 1.0 GB
    on 1.37.0 here, and 1.17 GB on 1.37.0 on CI's Linux runner. So the test
    reads the growth between half the source and all of it, which the
    read-ahead leaves out. Measured 2026-10-12 per GB of source: 0.43 GB on
    2.0.0 and 0.56 GB on 1.37.0, the binned data and the libraries' own
    structures; 1.7 GB on 1.36.1, whose `collect_batches` ran ahead of the
    build without limit and held the source."""
    half = _build_peak(big, lib, "parquet", "[0-2]")
    full = _build_peak(big, lib, "parquet")
    per_gb = (full - half) / 0.5e9
    assert per_gb < 1.0, (
        f"the peak grew by {per_gb:.2f} GB per GB of source "
        f"({half / 1e6:.0f} MB at 0.5 GB, {full / 1e6:.0f} MB at 1.0 GB)"
    )
