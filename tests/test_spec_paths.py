"""The spec builders on the paths no other test took (task 252): a spec's
name given by keyword, a target's name of the wrong type, a loss parameter
beside a loss that does not use it, a ``bin_edges`` table with a column that
is no feature, an old keyword beside its new name, and a duration that is
null. Each refusal names the spec and the parameter."""

from __future__ import annotations

import warnings
from datetime import timedelta

import polars as pl
import pytest

import polars_online as po

TIER = "essential"

BASE = dict(targets=["y"], features=["x"], half_life=10.0)


def test_the_name_may_be_given_by_keyword():
    """The name is the builders' first positional parameter; by keyword it
    is the same spec, and it is not read as a model parameter."""
    assert po.spec.ewridge(name="m", **BASE) == po.spec.ewridge("m", **BASE)


def test_a_target_s_name_must_be_a_string():
    with pytest.raises(TypeError, match="target 'y': name must be a str, got int"):
        po.target("y", name=3)  # type: ignore[arg-type]


def test_a_loss_parameter_beside_a_loss_that_does_not_use_it_is_refused():
    with pytest.raises(
        ValueError,
        match='spec "m": sgd huber_delta is for loss "huber"; loss "squared" does not use it',
    ):
        po.spec.sgd("m", loss="squared", huber_delta=1.0, **BASE)


def test_bin_edges_naming_a_column_that_is_no_feature_is_refused():
    with pytest.raises(
        ValueError, match=r"spec \"m\": bin_edges has \['z'\], which are not features"
    ):
        po.spec.marginal("m", bin_edges={"x": [0.0, 1.0], "z": [0.0, 1.0]}, **BASE)


def test_corrchange_s_old_keyword_beside_its_new_name_is_refused():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", po.PolarsOnlineDeprecationWarning)
        with pytest.raises(
            TypeError,
            match='spec "m": reset_on_flag was renamed change_action, and both are given: '
            "give change_action alone",
        ):
            po.spec.corrchange(
                "m", features=["x", "z"], half_life=10.0, reset_on_flag=True, change_action="flag"
            )


def test_a_null_duration_is_refused_by_name():
    with pytest.raises(ValueError, match='spec "m": half_life is a null duration'):
        po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            half_life=pl.lit(None, dtype=pl.Duration("us")),
            clock="t",
            gap_cap=10.0,
        )


def test_a_raw_spec_dict_with_a_duration_under_a_plain_parameter_is_refused():
    """A spec written as a dict may carry a duration where a clock parameter
    takes one, and a window expression in ``targets``; under any other key
    it is refused, naming the key, rather than serialized."""
    spec = po.spec.ewridge("m", **BASE)
    spec["model"]["ridge"] = timedelta(seconds=1)
    with pytest.raises(
        TypeError,
        match='spec "m": ridge takes no timedelta; a duration is a clock parameter\'s, and a '
        "window expression is a target's",
    ):
        po.ModelBank([spec])


def test_a_spec_missing_a_field_at_its_top_is_named_without_a_path():
    """serde names the field a spec lacks with its path inside the spec; a
    field of the spec itself has no path to give, and the message says the
    field alone."""
    spec = po.spec.ewridge("m", **BASE)
    del spec["name"]
    with pytest.raises(ValueError, match=r"^invalid spec: missing field `name`$"):
        po.spec.output_fields(spec)
