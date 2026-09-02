"""Every mistake a first-time user is likely to make, and what it says
(docs/IMPROVEMENTS.md U2). The messages name the spec, the parameter or the
column and its role, and where they can, the way out. A message that names a
JSON offset, a Rust type or an internal method is a regression here.
"""

from __future__ import annotations

import threading
import types
import typing

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import _spec

INF = float("inf")
BASE = dict(targets=["y"], features=["x0"], halflife=10.0)


def _df(n: int = 50) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [float(i) for i in range(n)],
            "x0": [float(i % 7) for i in range(n)],
            "y": [float((i % 7) * 2 + 1) for i in range(n)],
            "s": ["a"] * (n // 2) + ["b"] * (n - n // 2),
        }
    )


def _spec_dict(**kw) -> dict:
    return po.spec.ewridge("m", **{**BASE, **kw})


# --- the builders: wrong shapes are refused by parameter name ---------------

SHAPES = [
    (po.spec.ewridge, dict(targets="y"), "targets must be a list of strs, got str 'y'"),
    (po.spec.ewridge, dict(features="x0"), "features must be a list of strs, got str 'x0'"),
    (
        po.spec.ewridge,
        dict(halflife="10"),
        "halflife must be a number or a list of numbers, got str '10'",
    ),
    (
        po.spec.ewridge,
        dict(halflife=[10, "x"]),
        "halflife must be a number or a list of numbers, got list [10, 'x']",
    ),
    (po.spec.ewridge, dict(session_gap=[1]), "session_gap must be a number or a str, got list"),
    (po.spec.ewridge, dict(coef_every=1.5), "coef_every must be an int, got float 1.5"),
    (po.spec.ewridge, dict(standardize=1), "standardize must be a bool, got int 1"),
    (po.spec.ewridge, dict(coef0=[1.0, 2.0]), "coef0 must be a list of lists of numbers"),
    (
        po.spec.ewridge,
        dict(feature_sets=[("a", ["x0"])]),
        "feature_sets must be a dict of a str -> a list of strs",
    ),
    (po.spec.quantile, dict(quantile=[0.5]), "quantile must be a number, got list [0.5]"),
    (po.spec.lasso, dict(lasso_path=0.1), "lasso_path must be a list of numbers, got float"),
    (po.spec.ewridge, dict(bogus=1), "ewridge() got an unexpected keyword argument 'bogus'"),
]


@pytest.mark.parametrize("builder,kw,msg", SHAPES, ids=[m.split(" must")[0] for _, _, m in SHAPES])
def test_a_wrong_shape_names_the_parameter(builder, kw, msg):
    with pytest.raises(TypeError) as exc:
        builder("m", **{**BASE, **kw})
    assert msg in str(exc.value), str(exc.value)
    assert str(exc.value).startswith('spec "m": ')


def test_a_spec_name_must_be_a_string():
    with pytest.raises(TypeError, match="spec name must be a str, got int 1"):
        po.spec.ewridge(1, **BASE)


def test_ew_cov_has_no_targets():
    with pytest.raises(TypeError, match='spec "m": ew_cov\\(\\) takes no targets'):
        po.spec.ew_cov("m", features=["x0", "y"], targets=["y"], halflife=10.0)


VALUES = [
    (po.spec.ewridge, dict(coef_every=-1), "coef_every must be >= 0, got -1"),
    (po.spec.lasso, dict(lasso_path=[0.1], max_cd_iters=-1), "max_cd_iters must be >= 0"),
    (po.spec.ewridge, dict(solve_every=INF), "solve_every must be finite, got float inf"),
    (po.spec.ewridge, dict(resid_quantiles=[0.5, INF]), "resid_quantiles must be finite"),
    (po.spec.rls, dict(ridge=-INF), "ridge must be finite, got float -inf"),
]


@pytest.mark.parametrize("builder,kw,msg", VALUES, ids=[m.split(" must")[0] for _, _, m in VALUES])
def test_a_bad_value_names_the_parameter(builder, kw, msg):
    with pytest.raises(ValueError) as exc:
        builder("m", **{**BASE, **kw})
    assert msg in str(exc.value), str(exc.value)


def test_numpy_scalars_are_plain_numbers():
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0"], halflife=np.float64(10.0), coef_every=np.int64(3)
    )
    assert spec["halflife"] == 10.0 and spec["coef_every"] == 3
    out = po.ModelBank([spec]).fit_predict(_df())
    assert out["m"].struct.field("coef").null_count() < out.height


BUILDERS = {
    po.spec.ewridge: {},
    po.spec.rls: {},
    po.spec.lasso: dict(lasso_path=[0.1]),
    po.spec.kalman: dict(coef_halflife=10.0),
    po.spec.huber: {},
    po.spec.quantile: dict(quantile=0.5),
    po.spec.ftrl: {},
    po.spec.ew_cov: dict(features=["x0", "y"], targets=None),
    po.spec.sgd: {},
    po.spec.pa: {},
    po.spec.holt: dict(features=None),
}


def _members(hint) -> tuple:
    if typing.get_origin(hint) in (types.UnionType, typing.Union):
        return typing.get_args(hint)
    return (hint,)


def _inf_shaped_like(hint):
    """``"inf"`` in the shape the annotation asks for: bare for a float, nested
    once per ``list[...]`` otherwise (``q`` -> ``["inf"]``, ``coef0`` ->
    ``[["inf"]]``)."""
    members = _members(hint)
    if float in members:
        return "inf"
    for m in members:
        if typing.get_origin(m) is list:
            return [_inf_shaped_like(typing.get_args(m)[0])]
    return None


def _float_parameters(builder) -> dict[str, typing.Any]:
    hints = typing.get_type_hints(builder.__wrapped__) | typing.get_type_hints(_spec._common)
    skip = {"name", "model", "return", "targets", "features"}
    out = {}
    for key, hint in hints.items():
        if key not in skip and (inf := _inf_shaped_like(hint)) is not None:
            out[key] = inf
    return out


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_the_inf_table_matches_the_rust_side(builder):
    """``_INF_OK`` says which parameters Rust parses as ``Num`` (``inf``
    allowed) rather than ``f64``. Feed ``"inf"`` straight to Rust for every
    float parameter: an allowed one must get past the *parser* (a validation
    refusal is fine), and a refused one must be refused by Rust too, or the
    Python check is inventing a rule."""
    allowed = _spec._INF_OK["*"] | _spec._INF_OK.get(builder.__name__, frozenset())
    kwargs = {k: v for k, v in {**BASE, **BUILDERS[builder]}.items() if v is not None}
    for key, inf in _float_parameters(builder).items():
        spec = builder("m", **kwargs)
        where = spec if key in spec else spec["model"]
        assert key in where, f"{builder.__name__}.{key} is not a key of the spec dict"
        where[key] = inf
        try:
            po.ModelBank([spec])
        except ValueError as e:
            if key in allowed:
                assert "invalid spec" not in str(e), (key, str(e))
        else:
            assert key in allowed, f"{builder.__name__}.{key} accepts inf but is not in _INF_OK"


# --- hand-built dicts: serde names the path, and the visitors say what fits --


def test_a_hand_built_dict_is_checked_by_path():
    base = dict(name="m", model={"type": "ew_ridge"}, targets=["y"], features=["x0"])
    for bad, msg in [
        (dict(targets="y"), '[0].targets: invalid type: string "y", expected a sequence'),
        (
            dict(halflife="10"),
            '[0].halflife: invalid value: string "10", expected a number or a list of numbers',
        ),
        (dict(halflife=[10, "x"]), '[0].halflife[1]: invalid value: string "x"'),
        (
            dict(halflife=10, session_gap=[1]),
            "[0].session_gap: invalid type: sequence, expected a gap in clock units",
        ),
        (dict(halflife=10, model={"type": "ew_rdige"}), "[0].model.type: unknown variant"),
        (dict(halflife=10, model={}), "[0].model: missing field `type`"),
    ]:
        with pytest.raises(ValueError) as exc:
            po.ModelBank([{**base, **bad}])
        assert str(exc.value).startswith("invalid spec: ")
        assert msg in str(exc.value), str(exc.value)


# --- the frame: columns are named with their role, dtypes are checked --------


@pytest.mark.parametrize(
    "role,kw",
    [
        ("feature", dict(features=["nope"])),
        ("target", dict(targets=["nope"])),
        ("clock", dict(clock="nope", max_dclock=5.0)),
        ("session", dict(session="nope", session_gap=1.0)),
        ("weight", dict(weight="nope")),
        ("group", dict(group="nope")),
    ],
)
def test_a_missing_column_names_the_spec_the_role_and_the_frame(role, kw):
    bank = po.ModelBank([_spec_dict(**kw)])
    with pytest.raises(ValueError) as exc:
        bank.fit_predict(_df())
    text = str(exc.value)
    assert f'spec "m": {role} column "nope" not found' in text, text
    assert 'the frame has columns ["t", "x0", "y", "s"]' in text, text


@pytest.mark.parametrize(
    "role,kw",
    [
        ("feature", dict(features=["s"])),
        ("target", dict(targets=["s"])),
        ("clock", dict(clock="s", max_dclock=5.0)),
        ("weight", dict(weight="s")),
    ],
)
def test_a_string_column_is_refused_not_cast_to_null(role, kw):
    """A non-strict cast of a String column is all null: every prediction null
    and nothing to say why."""
    bank = po.ModelBank([_spec_dict(**kw)])
    with pytest.raises(ValueError) as exc:
        bank.fit_predict(_df())
    text = str(exc.value)
    assert f'spec "m": {role} column "s" has dtype str; it must be numeric' in text, text
    assert 'pl.col("s").cast(pl.Float64)' in text


def test_a_nested_key_column_is_refused():
    df = _df().with_columns(l=pl.concat_list("x0"))
    with pytest.raises(ValueError, match='group column "l" has dtype list\\[f64\\], which cannot'):
        po.ModelBank([_spec_dict(group="l")]).fit_predict(df)


def test_boolean_features_and_integer_keys_are_fine():
    df = _df().with_columns(b=pl.col("x0") > 3, g=(pl.col("x0") > 3).cast(pl.Int32))
    spec = _spec_dict(features=["x0", "b"], group="g", session="g", session_gap=1.0)
    out = po.ModelBank([spec]).fit_predict(df)
    assert out["m"].struct.field("pred_y").drop_nulls().len() > 0


def test_a_spec_named_like_an_input_column_is_refused():
    """Outputs are attached with `with_column`, which would replace the input."""
    with pytest.raises(ValueError) as exc:
        po.ModelBank([po.spec.ewridge("y", **BASE)]).fit_predict(_df())
    assert 'spec "y" has the same name as an input column' in str(exc.value)
    assert "Rename the spec" in str(exc.value)


def test_a_lazyframe_is_told_to_collect():
    bank = po.ModelBank([_spec_dict()])
    with pytest.raises(TypeError, match=r"not a LazyFrame: collect it first \(lf\.collect\(\)\)"):
        bank.fit_predict(_df().lazy())
    with pytest.raises(TypeError, match="takes a polars DataFrame, got dict"):
        bank.fit_predict({"y": [1.0]})


def test_concurrent_fit_predict_says_so():
    """The GIL is released for the run, so a second thread *can* reach the
    bank; it is refused with a sentence, not pyo3's "Already borrowed"."""
    n = 200_000
    df = pl.DataFrame(
        {
            "x0": np.random.default_rng(0).standard_normal(n),
            "y": np.random.default_rng(1).standard_normal(n),
        }
    )
    bank = po.ModelBank([_spec_dict(halflife=[10.0, 100.0, 1000.0], coef_every=1)])
    start = threading.Barrier(4)
    errors: list[BaseException] = []
    done: list[int] = []

    def go():
        start.wait()
        try:
            done.append(bank.fit_predict(df).height)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert done, errors
    assert errors, "four threads never overlapped; make the chunk bigger"
    for e in errors:
        assert isinstance(e, RuntimeError), e
        assert "running fit_predict on another thread" in str(e), str(e)
        assert "one ordered stream" in str(e)
    # The bank is intact: the winners' rows are counted, nothing else happened.
    assert bank.fit_predict(df.head(10)).height == 10
