"""The 1.x names that were read under an old spelling with a warning are
refused, naming the current spelling, where the code can know it.

2.0 keeps none of the forwards 1.x had: a builder's keyword, a spec dict's
key, a TOML file's key, a word a parameter took, a function, a method and a
call form each fail with an error that names what replaced them. The
warning class stays, for a name that is read under an old spelling in a
later release (``tests/test_renames.py`` holds the earlier refusals).
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest

import rheon as po
from rheon import _warnings

TIER = "essential"


def test_the_warning_is_a_deprecation_warning_and_exported():
    assert issubclass(po.PolarsOnlineDeprecationWarning, DeprecationWarning)
    assert "PolarsOnlineDeprecationWarning" in po.__all__


def test_no_forwarding_table_is_left():
    for name in ("_DEPRECATED", "_DEPRECATED_IN", "_DEPRECATED_WORDS_IN", "forward_deprecated"):
        assert not hasattr(_warnings, name), name


@pytest.mark.parametrize(
    ("builder", "kwargs", "old", "new"),
    [
        ("kmeans", {"k": 2}, "k", "n_clusters"),
        ("kmeans", {"n_clusters": 2, "seed_rule": "lloyd"}, "seed_rule", "init"),
        ("hmm", {"k": 2}, "k", "n_components"),
        ("hmm", {"n_components": 2, "covariance": "tied"}, "covariance", "covariance_type"),
        ("ew_class", {"classes": ["a", "b"], "window_every": 5}, "window_every", "snapshot_every"),
        ("ewridge", {"window_every": 5}, "window_every", "snapshot_every"),
        ("lasso", {"lasso_path": [0.1]}, "lasso_path", "penalties"),
        ("ew_cov", {"max_rows_between_pca": 5}, "max_rows_between_pca", "max_rows_between_pcas"),
        ("bocpd", {"emission": "robust"}, "emission", "covariance_type"),
        ("bocpd", {"prior_nu": 3.0}, "prior_nu", "degrees_of_freedom_prior"),
        ("corrchange", {"span_rows": 10}, "span_rows", "window_rows"),
        ("corrchange", {"reset_on_flag": True}, "reset_on_flag", "change_action"),
    ],
)
def test_a_renamed_builder_keyword_is_refused_naming_the_new_one(builder, kwargs, old, new):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=f"{old} was renamed {new}"):
            getattr(po.spec, builder)("m", features=["x0", "x1"], half_life=10.0, **kwargs)


@pytest.mark.parametrize(
    ("builder", "kwargs", "old", "new"),
    [
        ("kmeans", {"n_clusters": 2, "init": "kmeanspp"}, "kmeanspp", "k-means++"),
        ("hmm", {"n_components": 2, "covariance_type": "shared"}, "shared", "tied"),
        ("hmm", {"n_components": 2, "covariance_type": "diagonal"}, "diagonal", "diag"),
    ],
)
def test_a_renamed_word_is_refused_naming_the_new_one(builder, kwargs, old, new):
    with pytest.raises(ValueError, match=re.escape(f'the old name of "{new}"')):
        getattr(po.spec, builder)("m", features=["x0", "x1"], half_life=10.0, **kwargs)


def test_a_spec_dict_with_an_old_key_is_refused_naming_the_new_one():
    spec = po.spec.kmeans("m", features=["x0", "x1"], half_life=10.0, n_clusters=2)
    old = dict(spec, model=dict(spec["model"]))
    old["model"]["k"] = old["model"].pop("n_clusters")
    with pytest.raises(ValueError, match="k was renamed n_clusters"):
        po.ModelBank([old])


def test_another_models_key_of_the_same_name_is_plain():
    with pytest.raises(TypeError, match='ewridge\\(\\) takes no parameter "k"'):
        po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0, k=2)


def test_the_removed_functions_and_calls_refuse_naming_the_replacement():
    with pytest.raises(TypeError, match="po.ewm_mean"):
        po.rewm_mean("x", half_life=1.0, window_size=2.0)
    with pytest.raises(TypeError, match="po.gram.correlation"):
        po.corr.matrix([[1.0]])
    bank = po.ModelBank([po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0)])
    with pytest.raises(AttributeError, match="write_json"):
        bank.to_json()
    with pytest.raises(AttributeError, match="write_json"):
        bank.save_json("x.json")
    with pytest.raises(TypeError, match="rows_fed: it is a property"):
        bank.rows_fed()
    with pytest.raises(TypeError, match="output_fields: it is a property"):
        bank.output_fields()
    with pytest.raises(TypeError, match="path was renamed file"):
        bank.save(path="x.bin")
    with pytest.raises(TypeError, match="path was renamed source"):
        po.ModelBank.load(path="x.bin")


def test_the_removed_keywords_of_the_functions_refuse_naming_the_new_one():
    with pytest.raises(TypeError, match="spec_name was merged into spec"):
        po.eval.metrics(pl.DataFrame({"m": [1.0]}), spec_name="m")
    with pytest.raises(TypeError, match="spec_names was merged into specs"):
        po.eval.compare_specs(pl.DataFrame({"m": [1.0]}), spec_names=["m"])


def test_a_numeric_session_gap_is_refused():
    df = pl.DataFrame({"t": [0.0, 1.0], "s": [0, 0], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="session_gap = .* is no longer read"):
        po.stream.with_windows(
            df,
            f=po.ewm_mean("x", half_life=2.0),
            clock="t",
            gap_cap=10.0,
            session="s",
            session_gap=5.0,
        )


def test_the_exception_classes_have_the_one_parent_their_problem_names():
    assert po.WrongTypeError.__bases__ == (TypeError,)
    assert po.RefusedValueError.__bases__ == (ValueError,)
    assert po.OutputColumnNotFoundError.__bases__ == (po.ColumnNotFoundError,)
    assert not issubclass(po.OutputColumnNotFoundError, KeyError)
