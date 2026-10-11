"""The builders' keywords as they stand at 1.0: each default said in the
signature, no keyword a model cannot use, one name per concept across the
models, and a spec dict told how its keys map to a builder's keywords.
"""

from __future__ import annotations

import inspect
import json

import polars as pl
import pytest

import polars_online as po
from polars_online import _polars_online as native

TIER = "essential"


# --- P-5: kmeans and micro say their standardize default ------------------


@pytest.mark.parametrize(
    ("builder", "kw"),
    [(po.spec.kmeans, {"k": 2}), (po.spec.micro, {"eps": 0.3})],
    ids=["kmeans", "micro"],
)
def test_standardize_says_its_default(builder, kw):
    """``True`` in the signature and in the dict, where ``None`` stood for
    it and a reader had to look the default up."""
    assert inspect.signature(builder).parameters["standardize"].default is True
    spec = builder("m", features=["x0", "x1"], half_life=50.0, **kw)
    assert spec["model"]["standardize"] is True


@pytest.mark.parametrize(
    ("builder", "kw"),
    [(po.spec.kmeans, {"k": 2}), (po.spec.micro, {"eps": 0.3})],
    ids=["kmeans", "micro"],
)
def test_scale_floor_beside_standardize_false_is_refused(builder, kw):
    """``scale_floor`` floors the variance ``standardize`` measures in; with
    it off there is none, and the floor was taken and ignored."""
    with pytest.raises(ValueError, match="scale_floor floors the variance"):
        builder(
            "m", features=["x0", "x1"], half_life=50.0, standardize=False, scale_floor=0.1, **kw
        )
    spec = builder("m", features=["x0", "x1"], half_life=50.0, standardize=False, **kw)
    raw = dict(spec, model=dict(spec["model"], scale_floor=0.1))
    with pytest.raises(ValueError, match=r'^spec "m": scale_floor floors the variance'):
        native.validate_spec(json.dumps(raw))


# --- P-6: holt and seqtest have no features keyword -----------------------


@pytest.mark.parametrize(
    ("builder", "kw"),
    [(po.spec.holt, {}), (po.spec.seqtest, {"half_life": None})],
    ids=["holt", "seqtest"],
)
def test_a_model_with_no_features_has_no_features_keyword(builder, kw):
    """The keyword took only ``[]`` or ``None``: it is gone from the
    signature, and a call that gives it is told what the model reads."""
    assert "features" not in inspect.signature(builder).parameters
    with pytest.raises(TypeError, match=r"\(\) takes no features; it reads only its targets"):
        builder("m", targets=["y"], features=[], **kw)
    # A spec dict, which has every key, still says `features = []`.
    spec = builder("m", targets=["y"], **kw)
    assert spec["features"] == []
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    po.ModelBank([spec]).fit_predict(df)
