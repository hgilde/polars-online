"""Refusals that once came from Python or Polars without naming the
parameter, or under the wrong class, or not at all (task 359): each now
names the call and the parameter, in the class the problem calls for
(docs/WRITING.md, "Errors and warnings"), and a lone string given to
``drop_groups`` is one key, as Polars reads one name where it takes a list."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import corr

TIER = "essential"

N = 200


def _frame() -> pl.DataFrame:
    rng = np.random.default_rng(359)
    x0, x1 = rng.normal(size=N), rng.normal(size=N)
    return pl.DataFrame(
        {
            "t": np.arange(N, dtype=float),
            "g": ["ga", "gb"] * (N // 2),
            "x0": x0,
            "x1": x1,
            "y": 2.0 * x0 + x1 + rng.normal(size=N),
        }
    )


def _spec(name: str = "m", **kw: object) -> dict:
    kw.setdefault("half_life", 50.0)
    return po.spec.ewridge(name, targets=["y"], features=["x0", "x1"], **kw)


@pytest.fixture(scope="module")
def scored() -> pl.DataFrame:
    """Two specs' outputs, for the scoring functions."""
    bank = po.ModelBank([_spec("m"), _spec("m2", half_life=5.0)])
    return bank.fit_predict(_frame())


@pytest.fixture(scope="module")
def gram() -> dict:
    bank = po.ModelBank([_spec()])
    bank.fit_predict(_frame())
    return bank.gram("m")[0]


# --- chunk_size and closed_groups ---------------------------------------------


@pytest.mark.parametrize("size", [2.5, 2.0, "3", True])
def test_chunk_size_of_the_wrong_type_is_named(size: object) -> None:
    bank = po.ModelBank([_spec()])
    df = _frame()
    for who, call in [
        ("ModelBank.fit_predict_batches", lambda: bank.fit_predict_batches(df, chunk_size=size)),
        ("ModelBank.fit", lambda: bank.fit(df.lazy(), chunk_size=size)),
        ("df.online.fit_predict", lambda: df.online.fit_predict([_spec()], chunk_size=size)),
        ("lf.online.fit_predict", lambda: df.lazy().online.fit_predict([_spec()], chunk_size=size)),
    ]:
        with pytest.raises(po.WrongTypeError, match=f"^{who}: chunk_size must be a whole number"):
            call()


def test_chunk_size_takes_numpy_integers() -> None:
    out = list(po.ModelBank([_spec()]).fit_predict_batches(_frame(), chunk_size=np.int64(64)))
    assert [o.height for o in out] == [64, 64, 64, 8]


def test_closed_groups_that_is_not_a_path_is_named() -> None:
    bank = po.ModelBank([_spec()])
    with pytest.raises(TypeError, match="^ModelBank.fit_predict_batches: closed_groups must be"):
        bank.fit_predict_batches(_frame(), closed_groups=5)  # type: ignore[arg-type]


# --- skip_learned ----------------------------------------------------------------


def _learned_bank() -> po.ModelBank:
    bank = po.ModelBank([_spec(clock="t", gap_cap=10.0)])
    bank.fit_predict(_frame().head(50))
    return bank


def test_skip_learned_names_a_frame_of_the_wrong_type() -> None:
    with pytest.raises(TypeError, match="^ModelBank.skip_learned: frame must be a DataFrame"):
        _learned_bank().skip_learned({"t": [1.0]})  # type: ignore[call-overload]


def test_skip_learned_refuses_a_clock_of_the_wrong_type_as_a_type_error() -> None:
    bank = _learned_bank()
    text = _frame().with_columns(pl.col("t").cast(pl.String))
    with pytest.raises(po.WrongTypeError, match="which is not a clock"):
        bank.skip_learned(text)
    stamped = _frame().with_columns(pl.col("t").cast(pl.Int64).cast(pl.Datetime("us")))
    with pytest.raises(po.WrongTypeError, match="is temporal in the frame, but it was numeric"):
        bank.skip_learned(stamped)


# --- drop_groups -------------------------------------------------------------------


def test_drop_groups_reads_a_lone_string_as_one_key() -> None:
    bank = po.ModelBank([_spec(group="g")])
    bank.fit_predict(_frame())
    assert bank.drop_groups("ga") == 1  # not the keys "g" and "a"
    assert bank.groups()["group"].to_list() == ["gb"]
    assert bank.drop_groups(["gb"]) == 1


# --- po.stream.embargo -----------------------------------------------------------


@pytest.mark.parametrize("delay", [None, [1.0], True])
def test_embargo_names_a_delay_of_the_wrong_type(delay: object) -> None:
    with pytest.raises(po.WrongTypeError, match="^po.stream.embargo: delay must be a number"):
        po.stream.embargo(_frame(), clock="t", delay=delay)  # type: ignore[arg-type]


# --- po.eval -----------------------------------------------------------------------


@pytest.mark.parametrize("who", ["diebold_mariano", "clark_west"])
def test_lags_that_is_not_whole_is_named(scored: pl.DataFrame, who: str) -> None:
    names = {"a": "m", "b": "m2"} if who == "diebold_mariano" else {"big": "m", "small": "m2"}
    fn = getattr(po.eval, who)
    with pytest.raises(po.WrongTypeError, match=f"^po.eval.{who}: lags must be a whole number"):
        fn(scored, lags=1.5, **names)
    assert fn(scored, lags=np.int64(2), **names).height == 1


@pytest.mark.parametrize(
    ("who", "role", "call"),
    [
        ("metrics", "group", lambda df: po.eval.metrics(df, "m", group="nope")),
        ("sums", "group", lambda df: po.eval.sums(df, "m", group="nope")),
        ("sums", "weight", lambda df: po.eval.sums(df, "m", weight="nope")),
        (
            "window_metrics",
            "clock",
            lambda df: po.eval.window_metrics(df, "m", clock="nope", every=50.0),
        ),
        (
            "window_metrics",
            "group",
            lambda df: po.eval.window_metrics(df, "m", clock="t", every=50.0, group="nope"),
        ),
        ("seqtest", "group", lambda df: po.eval.seqtest(df, a="m", b="m2", group="nope")),
        (
            "diebold_mariano",
            "group",
            lambda df: po.eval.diebold_mariano(df, a="m", b="m2", group="nope"),
        ),
        (
            "clark_west",
            "group",
            lambda df: po.eval.clark_west(df, big="m", small="m2", group=["nope"]),
        ),
    ],
)
def test_eval_names_a_missing_column_as_this_librarys_class(
    scored: pl.DataFrame, who: str, role: str, call: object
) -> None:
    with pytest.raises(
        po.ColumnNotFoundError, match=f"^po.eval.{who}: {role} column 'nope' is not in the frame"
    ) as e:
        call(scored)  # type: ignore[operator]
    assert isinstance(e.value, ValueError)


# --- po.gbdt -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def parquet(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("gbdt") / "rows.parquet"
    _frame().drop("g").write_parquet(path)
    return str(path)


@pytest.mark.parametrize(
    ("seed", "kind", "match"),
    [
        (1.5, po.WrongTypeError, "seed must be a whole number"),
        ("1", po.WrongTypeError, "seed must be a whole number"),
        (-1, ValueError, "seed must be from 0 to 2\\*\\*64 - 1, but it is -1"),
        (2**64, ValueError, "seed must be from 0 to 2\\*\\*64 - 1"),
    ],
)
def test_xgboost_matrix_names_a_bad_seed(
    parquet: str, seed: object, kind: type, match: str
) -> None:
    with pytest.raises(kind, match=f"^po.gbdt.xgboost_matrix: {match}"):
        po.gbdt.xgboost_matrix(parquet, label="y", features=["x0"], seed=seed)  # type: ignore[arg-type]


def test_xgboost_matrix_takes_a_numpy_seed(parquet: str) -> None:
    m = po.gbdt.xgboost_matrix(parquet, label="y", features=["x0"], seed=np.uint64(2**64 - 1))
    assert m.num_row() == N


@pytest.mark.parametrize("ext", ["parquet", "arrow"])
def test_a_glob_that_matches_nothing_is_file_not_found_for_either_format(
    tmp_path: object, ext: str
) -> None:
    pattern = f"{tmp_path}/none*.{ext}"
    for who, call in [
        ("xgboost_matrix", lambda: po.gbdt.xgboost_matrix(pattern, label="y")),
        ("lightgbm_dataset", lambda: po.gbdt.lightgbm_dataset(pattern, label="y")),
    ]:
        with pytest.raises(FileNotFoundError, match=f"^po.gbdt.{who}: no file matches"):
            call()


# --- po.corr and po.gram -----------------------------------------------------------

R3 = np.array([[1.0, 0.8, 0.7], [0.8, 1.0, 0.75], [0.7, 0.75, 1.0]])


@pytest.mark.parametrize(("k", "kind"), [(0, ValueError), (-1, ValueError), (1.0, TypeError)])
def test_spectral_refuses_k_below_one(k: object, kind: type) -> None:
    with pytest.raises(kind, match="^po.corr.spectral: k must be"):
        corr.spectral(R3, k)  # type: ignore[arg-type]
    assert corr.spectral(R3, np.int32(1))[0].shape == (1,)


def test_fisher_se_checks_before_its_floor() -> None:
    with pytest.raises(ValueError, match="phi_a and phi_b must be given together"):
        corr.fisher_se(3, phi_a=0.5)
    with pytest.raises(ValueError, match="the product of phi_a and phi_b"):
        corr.fisher_se(2, phi_a=1.0, phi_b=1.0)
    assert np.isnan(corr.fisher_se(3, rho=0.5, phi_a=0.5, phi_b=0.5))


@pytest.mark.parametrize("rho", [1.5, -1.01, float("nan")])
def test_fisher_se_refuses_rho_outside_minus_one_to_one(rho: float) -> None:
    with pytest.raises(ValueError, match="^po.corr.fisher_se: rho must be a correlation"):
        corr.fisher_se(103, rho=rho)
    assert corr.fisher_se(103, rho=1.0) == 0.0


def test_from_blocks_refuses_an_asymmetric_table() -> None:
    with pytest.raises(ValueError, match=r"^po.corr.from_blocks: b must be symmetric, but b\["):
        corr.from_blocks(np.array([[0.6, 0.1], [0.2, 0.5]]), ["a", "a", "b", "b"])


def test_from_blocks_takes_what_block_means_returns() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(500, 9)) + rng.normal(size=(500, 1))
    r = np.corrcoef(x, rowvar=False)
    labels = ["a", "b", "c"] * 3
    b, _ = corr.block_means(r, labels)
    assert corr.from_blocks(b, labels).shape == (9, 9)
    single = corr.block_means(r, ["a", "b", "b", "b", "b", "b", "b", "b", "b"])[0]
    assert np.isnan(single[0, 0])  # a block of one column: nan matches nan
    assert corr.from_blocks(single, ["a", "b", "b", "b", "b", "b", "b", "b", "b"]).shape == (9, 9)


@pytest.mark.parametrize("columns", [["x0", "x0"], ["x0", 1], [1, -2]])
def test_gram_subset_and_condition_refuse_a_column_named_twice(gram: dict, columns: list) -> None:
    with pytest.raises(ValueError, match="^po.gram.subset: columns names column 'x0' twice"):
        po.gram.subset(gram, columns)
    with pytest.raises(ValueError, match="^po.gram.condition: features names column 'x0' twice"):
        po.gram.condition(gram, features=columns)
