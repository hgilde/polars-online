"""``po.gbdt``'s margin: ``base_margin=`` on ``xgboost_matrix`` and
``init_score=`` on ``lightgbm_dataset`` (docs/PLAN.md task 343,
docs/records/TREE-FEATURES.md §4).

Held to each library's own in-memory route with the same margin by the hash
of the trees it grows, as ``tests/test_gbdt.py`` holds the rest: for a
regression, a binary and a three-class objective, through every route
``po.gbdt`` has (both engines, LightGBM's two-pass and in-memory routes,
the native engine's Arrow IPC reader, Parquet files, a query, a frame in
many chunks), each at many batch boundaries (``_CHUNK`` is cut to 999 rows
here, so a Polars source arrives in batches of at most 999 and the Arrow
files hold batches of 1,000). Each margin moves the trees: the hash with it
is not the hash without it, so a margin dropped in the push fails every
comparison. A validation set's own margin is held the same way, through
early stopping's record, and each refusal by its message.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from polars_online import gbdt
from test_gbdt import FEATURES, LGB_DATA, TYPES, frame, lgb_trees, numpy_x, xgb_trees

TIER = "essential"

N = 6_000
#: Each objective: its label, its margin column, the XGBoost and LightGBM
#: parameters that read a margin of that width.
OBJECTIVES = {
    "regression": ("y", "m", {"objective": "reg:squarederror"}, {"objective": "regression"}),
    "binary": ("b", "m", {"objective": "binary:logistic"}, {"objective": "binary"}),
    "three classes, a List": (
        "c",
        "ml",
        {"objective": "multi:softprob", "num_class": 3},
        {"objective": "multiclass", "num_class": 3},
    ),
    "three classes, an Array": (
        "c",
        "ma",
        {"objective": "multi:softprob", "num_class": 3},
        {"objective": "multiclass", "num_class": 3},
    ),
}
XGB = {"tree_method": "hist", "max_depth": 3, "eta": 0.3, "max_bin": 256, "nthread": 4, "seed": 0}
LGB = {
    "num_leaves": 7,
    "learning_rate": 0.3,
    "num_threads": 4,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}


def with_margins(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """``test_gbdt.frame``'s columns, a level that random-walks and no
    feature carries added to the target, a binary and a three-class label,
    and margins: ``m`` the level (one value a row), ``ml`` and ``ma`` three
    values a row as a ``List`` and an ``Array``, ``z`` a margin of 0."""
    rng = np.random.default_rng(seed)
    n = df.height
    level = np.cumsum(rng.normal(scale=0.05, size=n))
    y = df["y"].to_numpy() + level
    three = rng.normal(scale=0.5, size=(n, 3)) + level[:, None] * np.array([1.0, 0.0, -1.0])
    return df.with_columns(
        y=pl.Series(y),
        b=pl.Series((y > np.median(y)).astype(float)),
        c=pl.Series(np.digitize(y - level, np.quantile(y - level, [1 / 3, 2 / 3])).astype(float)),
        m=pl.Series(level),
        ml=pl.Series(three.tolist(), dtype=pl.List(pl.Float64)),
        ma=pl.Series(three.tolist(), dtype=pl.Array(pl.Float64, 3)),
        z=pl.lit(0.0),
    )


def margin_of(df: pl.DataFrame, column: str) -> np.ndarray:
    """The margin as each library's numpy route takes it: ``n`` values, or
    ``n`` rows of ``k``."""
    s = df[column]
    if isinstance(s.dtype, (pl.List, pl.Array)):
        return np.array(s.to_list(), dtype=np.float64)
    return s.to_numpy()


@pytest.fixture(autouse=True)
def small_batches(monkeypatch):
    monkeypatch.setattr(gbdt, "_CHUNK", 999)


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return with_margins(frame(N), 0)


@pytest.fixture(scope="module")
def valid() -> pl.DataFrame:
    return with_margins(frame(2_000, seed=1), 1)


@pytest.fixture(scope="module")
def sources(tmp_path_factory, data) -> dict[str, object]:
    """The data as a frame, in many chunks, as Arrow IPC files of several
    record batches each, as Parquet files and as a query over them."""
    d = tmp_path_factory.mktemp("margin")
    ipc, parquet = [], []
    for k, part in enumerate(data.iter_slices(2_500)):
        chunked = pl.concat(list(part.iter_slices(1_000)), rechunk=False)
        p = d / f"part-{k}.arrow"
        chunked.write_ipc(p)
        ipc.append(str(p))
        q = d / f"part-{k}.parquet"
        part.write_parquet(q, row_group_size=700)
        parquet.append(str(q))
    chunked = pl.concat(list(data.iter_slices(777)), rechunk=False)
    assert chunked.n_chunks() > 5
    return {
        "frame": data,
        "chunked": chunked,
        "ipc": ipc,
        "parquet": parquet,
        "query": pl.scan_parquet(parquet).with_columns(pl.col("x1") * 1.0),
    }


def xgb_own(df: pl.DataFrame, label: str, margin: str | None, ref=None) -> xgb.QuantileDMatrix:
    return xgb.QuantileDMatrix(
        numpy_x(df),
        label=df[label].to_numpy(),
        base_margin=None if margin is None else margin_of(df, margin),
        feature_names=FEATURES,
        feature_types=TYPES,
        enable_categorical=True,
        max_bin=256,
        ref=ref,
    )


def lgb_own(df: pl.DataFrame, label: str, margin: str | None, reference=None) -> lgb.Dataset:
    return lgb.Dataset(
        numpy_x(df, np.float32),
        label=df[label].to_numpy(),
        init_score=None if margin is None else margin_of(df, margin),
        feature_name=FEATURES,
        categorical_feature=[FEATURES.index("cat")],
        params={"max_bin": 255, **LGB_DATA},
        reference=reference,
        free_raw_data=False,
    ).construct()


@pytest.mark.parametrize("objective", list(OBJECTIVES))
@pytest.mark.parametrize("engine", ["native", "python"])
def test_xgboost_grows_its_own_trees_around_the_margin(data, sources, engine, objective):
    """Every source, both engines: the trees of XGBoost's numpy route with
    the same ``base_margin``, on the same cuts; and not the trees without
    the margin, so the margin reached XGBoost."""
    label, margin, params, _ = OBJECTIVES[objective]
    params = {**XGB, **params}
    without = gbdt.xgboost_matrix(data, label=label, features=FEATURES, engine=engine)
    plain = xgb_trees(xgb.train(params, without, 8))
    for name, source in sources.items():
        m = gbdt.xgboost_matrix(
            source, label=label, features=FEATURES, base_margin=margin, engine=engine
        )
        assert m.num_row() == data.height
        # XGBoost reads a margin back as float32.
        np.testing.assert_array_equal(
            m.get_base_margin().ravel(), margin_of(data, margin).ravel().astype(np.float32)
        )
        own = xgb_own(data, label, margin, ref=m)
        grown = xgb_trees(xgb.train(params, m, 8))
        assert grown == xgb_trees(xgb.train(params, own, 8)), name
        assert grown != plain, f"{name}: the margin moved no tree"


@pytest.mark.parametrize("objective", list(OBJECTIVES))
@pytest.mark.parametrize("in_memory", [False, True])
@pytest.mark.parametrize("engine", ["native", "python"])
def test_lightgbm_grows_its_own_trees_around_the_initial_scores(
    data, sources, engine, in_memory, objective
):
    """Every source, both engines, both routes: the trees of LightGBM's
    numpy route with the same ``init_score``, and not the trees without it;
    the scores stored class-major, as LightGBM's own route stores them."""
    label, margin, _, params = OBJECTIVES[objective]
    params = {**LGB, **params}
    kw = dict(label=label, features=FEATURES, params=LGB_DATA, engine=engine, in_memory=in_memory)
    plain = lgb_trees(lgb.train(params, gbdt.lightgbm_dataset(data, **kw), 8))
    own = lgb_own(data, label, margin)
    want = lgb_trees(lgb.train(params, own, 8))
    for name, source in sources.items():
        ds = gbdt.lightgbm_dataset(source, **kw, init_score=margin)
        assert ds.num_data() == data.height
        np.testing.assert_array_equal(
            np.asarray(ds.get_field("init_score")), np.asarray(own.get_field("init_score"))
        )
        grown = lgb_trees(lgb.train(params, ds, 8))
        assert grown == want, name
        assert grown != plain, f"{name}: the initial scores moved no tree"


@pytest.mark.parametrize("engine", ["native", "python"])
def test_a_validation_matrix_reads_its_own_margin(data, valid, engine):
    """Early stopping on a validation matrix with its own ``base_margin``
    keeps XGBoost's numpy route's record, and the record without margins
    keeps the numpy route's without them, which differs: the validation
    margin is the validation rows', not the training rows'."""
    params = {**XGB, "eta": 0.8, "max_depth": 6}
    records = {}
    for margin in ("m", None):
        train = gbdt.xgboost_matrix(
            data, label="y", features=FEATURES, base_margin=margin, engine=engine
        )
        vm = gbdt.xgboost_matrix(
            valid, label="y", features=FEATURES, base_margin=margin, ref=train, engine=engine
        )
        own = xgb_own(data, "y", margin, ref=train)
        own_v = xgb_own(valid, "y", margin, ref=own)
        rec_a: dict = {}
        rec_b: dict = {}
        for m, v, rec in ((train, vm, rec_a), (own, own_v, rec_b)):
            xgb.train(
                params,
                m,
                300,
                evals=[(v, "valid")],
                early_stopping_rounds=3,
                evals_result=rec,
                verbose_eval=False,
            )
        assert len(rec_a["valid"]["rmse"]) < 300, "early stopping stopped it"
        assert rec_a == rec_b, margin
        records[margin] = rec_a["valid"]["rmse"]
    assert records["m"][:5] != records[None][:5]
    # A prediction on the validation matrix adds its margin, as XGBoost's own does.
    train = gbdt.xgboost_matrix(data, label="y", features=FEATURES, base_margin="m")
    booster = xgb.train(XGB, train, 5)
    vm = gbdt.xgboost_matrix(
        valid, label="y", features=FEATURES, base_margin="m", ref=train, engine=engine
    )
    plain = xgb.DMatrix(
        numpy_x(valid),
        base_margin=valid["m"].to_numpy(),
        feature_names=FEATURES,
        feature_types=TYPES,
        enable_categorical=True,
    )
    np.testing.assert_array_equal(booster.predict(vm), booster.predict(plain))


@pytest.mark.parametrize("in_memory", [False, True])
@pytest.mark.parametrize("engine", ["native", "python"])
def test_a_validation_dataset_reads_its_own_initial_scores(data, valid, engine, in_memory):
    """As for XGBoost: early stopping keeps LightGBM's numpy route's record
    with the validation rows' own ``init_score``, and without them its
    record without, which differs."""
    params = {**LGB, "objective": "multiclass", "num_class": 3, "learning_rate": 0.9}
    records = {}
    for margin in ("ml", None):
        kw = dict(label="c", features=FEATURES, init_score=margin, engine=engine)
        train = gbdt.lightgbm_dataset(data, **kw, params=LGB_DATA, in_memory=in_memory)
        vd = gbdt.lightgbm_dataset(valid, **kw, reference=train, in_memory=in_memory)
        own = lgb_own(data, "c", margin)
        own_v = lgb_own(valid, "c", margin, reference=own)
        recs: list[dict] = [{}, {}]
        best = []
        for d, v, rec in ((train, vd, recs[0]), (own, own_v, recs[1])):
            b = lgb.train(
                params,
                d,
                300,
                valid_sets=[v],
                callbacks=[lgb.early_stopping(3, verbose=False), lgb.record_evaluation(rec)],
            )
            best.append(b.best_iteration)
        assert best[0] == best[1] < 299, "early stopping stopped it"
        a, b = recs[0]["valid_0"]["multi_logloss"], recs[1]["valid_0"]["multi_logloss"]
        np.testing.assert_allclose(a, b, rtol=1e-12)
        records[margin] = a
    assert records["ml"][0] != pytest.approx(records[None][0], rel=1e-6)


def test_a_margin_of_any_numeric_type_is_read_as_float64(data, tmp_path):
    """A margin stored as float32, as integers, or as a list of float32, in
    a frame and in an Arrow IPC file (which the native engine reads
    itself), is the same margin as its values in float64."""
    df = data.with_columns(
        m32=pl.col("m").cast(pl.Float32),
        mi=(pl.col("m") * 10).round().cast(pl.Int16),
        ml32=pl.col("ml").cast(pl.List(pl.Float32)),
    )
    path = tmp_path / "typed.arrow"
    df.write_ipc(path)
    params = {**XGB, "objective": "multi:softprob", "num_class": 3}
    for column, kw in (("m32", {}), ("mi", {}), ("ml32", params)):
        own = xgb_own(df, "c" if kw else "y", column)
        want = xgb_trees(xgb.train(kw or XGB, own, 5))
        for source in (df, str(path)):
            m = gbdt.xgboost_matrix(
                source, label="c" if kw else "y", features=FEATURES, base_margin=column, ref=own
            )
            assert xgb_trees(xgb.train(kw or XGB, m, 5)) == want, (column, source)
            ds = gbdt.lightgbm_dataset(source, label="y", features=FEATURES, init_score=column)
            # LightGBM hands back a K-class dataset's scores as n rows of K.
            np.testing.assert_array_equal(
                ds.get_field("init_score"), margin_of(df, column).astype(np.float64)
            )


def test_a_reference_built_by_the_library_is_read_for_its_margin(data, valid):
    """A training set built by the library's own route, with or without a
    margin, is read for it as one built here is."""
    own = xgb_own(data, "c", "ml")
    gbdt.xgboost_matrix(valid, label="c", features=FEATURES, base_margin="ml", ref=own)
    with pytest.raises(ValueError, match=r"ref was built with a base_margin of 3 values a row"):
        gbdt.xgboost_matrix(valid, label="c", features=FEATURES, ref=own)
    with pytest.raises(ValueError, match=r"the ref's margin holds 3 values a row"):
        gbdt.xgboost_matrix(valid, label="c", features=FEATURES, base_margin="m", ref=own)
    own = lgb_own(data, "y", None)
    with pytest.raises(ValueError, match=r"init_score= is given, and reference was built without"):
        gbdt.lightgbm_dataset(valid, label="y", features=FEATURES, init_score="m", reference=own)
    own = lgb_own(data, "y", "m")
    vd = gbdt.lightgbm_dataset(
        valid, label="y", features=FEATURES, init_score="m", reference=own, engine="native"
    )
    np.testing.assert_array_equal(vd.get_field("init_score"), valid["m"].to_numpy())


def _at(df: pl.DataFrame, column: str, row: int, value) -> pl.DataFrame:
    """``df`` with ``column``'s value at ``row`` replaced."""
    s = df[column].to_list()
    s[row] = value
    return df.with_columns(pl.Series(column, s, dtype=df[column].dtype))


@pytest.mark.parametrize("engine", ["native", "python"])
@pytest.mark.parametrize("lib", ["xgboost", "lightgbm"])
def test_a_margin_that_cannot_be_read_is_refused_by_name(data, valid, lib, engine):
    role = "base_margin" if lib == "xgboost" else "init_score"
    whose = "ref" if lib == "xgboost" else "reference"

    def build(df, margin, **kw):
        if lib == "xgboost":
            return gbdt.xgboost_matrix(
                df, label="y", features=FEATURES, base_margin=margin, engine=engine, **kw
            )
        return gbdt.lightgbm_dataset(
            df, label="y", features=FEATURES, init_score=margin, engine=engine, **kw
        )

    cases = [
        (_at(data, "m", 1234, None), "m", rf"the {role} column `m` has a null at row 1234"),
        (_at(data, "ml", 4321, [0.0, None, 1.0]), "ml", r"`ml` has a null at row 4321"),
        (_at(data, "ml", 4322, None), "ml", r"`ml` has a null at row 4322"),
        (_at(data, "m", 7, float("inf")), "m", r"`m` holds inf at row 7, and a margin must be"),
        (_at(data, "ml", 8, [0.0, float("nan"), 1.0]), "ml", r"`ml` holds NaN at row 8"),
        (_at(data, "ml", 2050, [0.0, 1.0]), "ml", r"holds 2 values at row 2050 and 3 at row 0"),
        (_at(data, "ml", 0, []), "ml", r"`ml` holds no values at row 0, and a margin needs one"),
    ]
    for df, margin, match in cases:
        with pytest.raises(ValueError, match=match):
            build(df, margin)
    with pytest.raises(ValueError, match=rf"the {role} column 'q' is not in the source"):
        build(data, "q")
    with pytest.raises(ValueError, match=rf"the {role} column 's' is String, and a margin is"):
        build(data.with_columns(s=pl.lit("a")), "s")
    with pytest.raises(ValueError, match=rf"the {role} column 'v' is List\(Boolean\)"):
        build(data.with_columns(v=pl.lit([True])), "v")
    with pytest.raises(ValueError, match=rf"'x0' is named as a feature and as the {role}"):
        build(data, "x0")
    with pytest.raises(ValueError, match=rf"'y' is named as the {role} and as the label"):
        build(data, "y")
    # Every column but the label and the margin is a feature by default.
    if lib == "xgboost":
        m = gbdt.xgboost_matrix(data.select(*FEATURES, "y", "m"), label="y", base_margin="m")
        assert m.feature_names == FEATURES
    train = build(data, "ml")
    with pytest.raises(ValueError, match=rf"{whose} was built with a {role} of 3 values a row"):
        build(valid, None, **{whose: train})
    with pytest.raises(
        ValueError, match=rf"`m` holds 1 value at row 0, and the {whose}'s margin holds 3 values"
    ):
        build(valid, "m", **{whose: train})
    plain = build(data, None)
    with pytest.raises(ValueError, match=rf"{role}= is given, and {whose} was built without one"):
        build(valid, "m", **{whose: plain})
