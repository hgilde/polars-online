"""The builders' keywords as they stand at 1.0: each default said in the
signature, no keyword a model cannot use, one name per concept across the
models, and a spec dict told how its keys map to a builder's keywords.
"""

from __future__ import annotations

import inspect
import json
import re

import polars as pl
import pytest

import polars_online as po
from polars_online import _polars_online as native

TIER = "essential"


# --- P-5: kmeans and micro say their standardize default ------------------


@pytest.mark.parametrize(
    ("builder", "kw"),
    [(po.spec.kmeans, {"n_clusters": 2}), (po.spec.micro, {"eps": 0.3})],
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
    [(po.spec.kmeans, {"n_clusters": 2}), (po.spec.micro, {"eps": 0.3})],
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
    [(po.spec.holt, {"half_life": 10.0}), (po.spec.seqtest, {"half_life": None})],
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


# --- P-2, P-3, P-7, P-8, P-9 and the folded renames: forwarded -------------

#: A minimal call of each builder whose keyword was renamed, by model type.
BASE: dict[str, dict[str, object]] = {
    "kmeans": {"features": ["x0", "x1"], "half_life": 50.0, "n_clusters": 2},
    "hmm": {"features": ["x0", "x1"], "half_life": 50.0, "n_components": 2, "precision_prior": 0.1},
    "ew_class": {
        "features": ["x0"],
        "label": "y",
        "classes": ["a", "b"],
        "precision_prior": 1.0,
        "half_life": 50.0,
    },
    "ewridge": {"targets": ["y"], "features": ["x0"], "half_life": 50.0},
    "lasso": {"targets": ["y"], "features": ["x0"], "half_life": 50.0, "penalties": [0.1, 0.0]},
    "ew_cov": {"features": ["x0", "x1"], "half_life": 50.0},
    "marginal": {"targets": ["y"], "features": ["x0"], "half_life": 50.0},
    "bocpd": {"features": ["x0", "x1"]},
    "corrchange": {"features": ["x0", "x1"], "window_rows": 20},
}

#: (model, the old keyword and its value, the new keyword and its value,
#: what else the call needs).
RENAMED = [
    ("kmeans", ("k", 3), ("n_clusters", 3), {}),
    ("hmm", ("k", 3), ("n_components", 3), {}),
    ("kmeans", ("seed_rule", "first"), ("init", "first"), {}),
    ("hmm", ("seed_rule", "first"), ("init", "first"), {"warm_rows": 10}),
    ("kmeans", ("seed_rule", "kmeanspp"), ("init", "k-means++"), {}),
    ("kmeans", ("init", "kmeanspp"), ("init", "k-means++"), {}),
    ("hmm", ("covariance", "shared"), ("covariance_type", "tied"), {}),
    ("hmm", ("covariance", "diagonal"), ("covariance_type", "diag"), {}),
    ("hmm", ("covariance_type", "shared"), ("covariance_type", "tied"), {}),
    ("ew_class", ("covariance", "shared"), ("covariance_type", "tied"), {}),
    ("ew_class", ("covariance", "full"), ("covariance_type", "full"), {}),
    ("ew_class", ("covariance_type", "diagonal"), ("covariance_type", "diag"), {}),
    ("ewridge", ("window_every", 5.0), ("snapshot_every", 5.0), {"window_size": 100.0}),
    ("lasso", ("window_every", 5.0), ("snapshot_every", 5.0), {"window_size": 100.0}),
    ("ew_cov", ("window_every", 5.0), ("snapshot_every", 5.0), {"window_size": 100.0}),
    ("ew_class", ("window_every", 5.0), ("snapshot_every", 5.0), {"window_size": 100.0}),
    ("marginal", ("window_every", 5.0), ("snapshot_every", 5.0), {"window_size": 100.0}),
    ("ew_cov", ("max_rows_between_pca", 10), ("max_rows_between_pcas", 10), {"pca": 1}),
    ("lasso", ("lasso_path", [0.2, 0.0]), ("penalties", [0.2, 0.0]), {}),
    ("bocpd", ("prior_mean", [0.0, 0.0]), ("mean_prior", [0.0, 0.0]), {}),
    ("bocpd", ("prior_kappa", 2.0), ("mean_precision_prior", 2.0), {}),
    ("bocpd", ("prior_nu", 5.0), ("degrees_of_freedom_prior", 5.0), {}),
    ("bocpd", ("prior_scale", [1.0]), ("covariance_prior", [1.0]), {}),
    ("bocpd", ("emission", "gaussian"), ("covariance_type", "full"), {}),
    ("bocpd", ("emission", "diag"), ("covariance_type", "diag"), {}),
    ("corrchange", ("span_rows", 30), ("window_rows", 30), {}),
    (
        "corrchange",
        ("reset_on_flag", True),
        ("change_action", "reset"),
        {"kind": "window"},
    ),
    ("corrchange", ("reset_on_flag", False), ("change_action", "flag"), {}),
]

#: The keys `BASE` spells in the new way that a renamed keyword replaces.
_REPLACES = {"k": ("n_clusters", "n_components"), "span_rows": ("window_rows",)}
_REPLACES |= {"lasso_path": ("penalties",)}


def _build(kind: str, kw: dict[str, object]) -> dict:
    base = {k: v for k, v in BASE[kind].items() if k not in kw}
    for old, news in _REPLACES.items():
        if old in kw:
            for new in news:
                base.pop(new, None)
    return getattr(po.spec, kind)("m", **base, **kw)


_IDS = [f"{k}.{o[0]}={o[1]}" for k, o, _, _ in RENAMED]


@pytest.mark.parametrize(("kind", "old", "new", "extra"), RENAMED, ids=_IDS)
def test_a_renamed_keyword_is_read_as_the_new_one_with_a_warning(kind, old, new, extra):
    """The old keyword, or the old word, still builds the same spec, with a
    ``PolarsOnlineDeprecationWarning`` naming what replaced it; from the next
    major version it is refused."""
    want = _build(kind, {new[0]: new[1], **extra})
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="deprecated") as caught:
        got = _build(kind, {old[0]: old[1], **extra})
    assert got == want
    said = " ".join(str(w.message) for w in caught)
    assert new[0] in said and "next major version" in said, said


@pytest.mark.parametrize(("kind", "old", "new", "extra"), RENAMED, ids=_IDS)
def test_a_spec_dict_with_an_old_key_is_read_as_the_new_one(kind, old, new, extra):
    """A hand-written dict is forwarded at the bank's door the same way,
    keyed by the model it is in."""
    want = _build(kind, {new[0]: new[1], **extra})
    model = dict(want["model"])
    model.pop(new[0])
    model[old[0]] = old[1]
    with pytest.warns(po.PolarsOnlineDeprecationWarning):
        bank = po.ModelBank([dict(want, model=model)])
    assert bank.specs == po.ModelBank([want]).specs


def test_a_tempered_emission_is_read_as_robust_beta():
    """``emission="robust"`` was the tempered per-feature predictive at a
    default of 0.1: it is ``covariance_type="diag"`` with ``robust_beta``,
    0.1 where none is given."""
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="robust_beta"):
        got = _build("bocpd", {"emission": "robust"})
    assert got == _build("bocpd", {"covariance_type": "diag", "robust_beta": 0.1})
    with pytest.warns(po.PolarsOnlineDeprecationWarning):
        given = _build("bocpd", {"emission": "robust", "robust_beta": 0.05})
    assert given["model"]["robust_beta"] == 0.05


def test_hmm_reads_seed_rule_kmeanspp_as_two_renames():
    """``seed_rule="kmeanspp"`` is a key and a word renamed at once, each
    warned."""
    with pytest.warns(po.PolarsOnlineDeprecationWarning) as caught:
        got = _build("hmm", {"seed_rule": "kmeanspp", "warm_rows": 10})
    assert got["model"]["init"] == "k-means++"
    assert len(caught) == 2


def test_an_old_name_is_read_only_in_its_own_model():
    """``k`` is ``kmeans``' and ``hmm``'s old name, and nothing to another
    model: an ``ewridge`` table naming it is refused as an unknown key, and a
    feature called ``k`` in a table keyed by feature stays a feature."""
    spec = po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=50.0)
    raw = dict(spec, model={**spec["model"], "k": 2})
    with pytest.raises(ValueError, match="unknown field `k`"):
        po.ModelBank([raw])
    kept = po.spec.marginal(
        "m", targets=["y"], features=["k"], half_life=50.0, bin_edges={"k": [0.0]}
    )
    assert po.ModelBank([kept]).specs[0]["model"]["bin_edges"] == kept["model"]["bin_edges"]


def test_an_old_name_beside_its_new_one_is_refused_naming_both():
    with pytest.raises(TypeError, match="k was renamed n_clusters, and both are given"):
        po.spec.kmeans("m", features=["x0"], half_life=50.0, k=2, n_clusters=2)
    with pytest.raises(TypeError, match="emission was renamed covariance_type, and both"):
        po.spec.bocpd("m", features=["x0"], emission="gaussian", covariance_type="full")


def test_the_new_words_are_scikit_learns():
    """``covariance_type`` takes ``GaussianMixture``'s words, ``init``
    ``KMeans``' spelling of k-means++, and ``ew_class`` says its default."""
    assert inspect.signature(po.spec.ew_class).parameters["covariance_type"].default == "full"
    for kind, extra in [("kmeans", {}), ("hmm", {"warm_rows": 10})]:
        assert _build(kind, {"init": "k-means++", **extra})["model"]["init"] == "k-means++"
    for shape in ("full", "tied", "diag"):
        _build("hmm", {"covariance_type": shape})
        _build("ew_class", {"covariance_type": shape})
    with pytest.raises(ValueError, match='covariance_type must be one of "full", "diag"'):
        _build("bocpd", {"covariance_type": "tied"})
    with pytest.raises(ValueError, match="robust_beta tempers the per-feature predictive"):
        _build("bocpd", {"covariance_type": "full", "robust_beta": 0.1})


# --- C-7: a table says targets and features ----------------------------------


@pytest.mark.parametrize(
    ("spec", "key", "says"),
    [
        (
            {
                "name": "m",
                "model": {"type": "ew_class", "classes": ["a", "b"], "precision_prior": 1.0},
                "label": "y",
                "features": ["x0"],
                "half_life": 10.0,
            },
            "label",
            'is ew_class\'s builder keyword; a spec table says `targets = ["<the label column>"]`',
        ),
        (
            {"name": "m", "model": {"type": "audit"}, "columns": ["x0", "x1"]},
            "columns",
            "is audit's builder keyword; a spec table says `features = [...]`",
        ),
    ],
    ids=["ew_class.label", "audit.columns"],
)
def test_a_table_with_a_builder_keyword_is_told_the_tables_key(spec, key, says):
    """``ew_class(label=)`` and ``audit(columns=)`` name their columns for
    what they are to the model; a spec dict or a TOML table names them
    ``targets`` and ``features`` as for every model, and one that says the
    builder's word is told which key to use, where it read only "unknown
    field"."""
    with pytest.raises(ValueError, match=re.escape(f"`{key}` {says}")):
        po.ModelBank([spec])
    built = (
        po.spec.ew_class(
            "m", features=["x0"], label="y", classes=["a", "b"], precision_prior=1.0, half_life=10.0
        )
        if key == "label"
        else po.spec.audit("m", columns=["x0", "x1"])
    )
    assert key not in built and ("targets" if key == "label" else "features") in built
