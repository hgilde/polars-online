"""Every mistake a first-time user is likely to make, and what it says
(docs/IMPROVEMENTS.md U2). The messages name the spec, the parameter or the
column and its role, and where they can, the way out. A message that names a
JSON offset, a Rust type or an internal method is a regression here.
"""

from __future__ import annotations

import datetime
import math
import re
import threading
import types
import typing

import numpy as np
import polars as pl
import pytest

import polars_online as po
from conftest import run_online
from polars_online import _spec

INF = float("inf")
BASE = dict(targets=["y"], features=["x0"], half_life=10.0)


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
    (
        po.spec.ewridge,
        dict(targets="y"),
        "targets must be a list of strs, po.target tables or window expressions "
        "looking ahead, got str 'y'",
    ),
    (po.spec.ewridge, dict(features="x0"), "features must be a list of strs, got str 'x0'"),
    # A string is a duration's shape now ("10m", task 88), so a wrong
    # shape for a clock parameter is anything that is neither.
    (
        po.spec.ewridge,
        dict(half_life=True),
        "half_life must be a number or a duration or a list of numbers or durations, got bool",
    ),
    (
        po.spec.ewridge,
        dict(half_life=[10, None]),
        "half_life must be a number or a duration or a list of numbers or durations, got list",
    ),
    (
        po.spec.ewridge,
        dict(session_gap=[1]),
        "session_gap must be a number or a duration, got list",
    ),
    (
        po.spec.ewridge,
        dict(max_rows_between_coefs=1.5),
        "max_rows_between_coefs must be an int, got float 1.5",
    ),
    (po.spec.ewridge, dict(standardize=1), "standardize must be a bool, got int 1"),
    (po.spec.ewridge, dict(coef_prior=[1.0, 2.0]), "coef_prior must be a list of lists of numbers"),
    (
        po.spec.ewridge,
        dict(feature_sets=[("a", ["x0"])]),
        "feature_sets must be a dict of a str -> a list of strs",
    ),
    (po.spec.quantile, dict(quantile=[0.5]), "quantile must be a number, got list [0.5]"),
    # A table target is a TypedDict, which the check did not read: a wrong
    # entry reached Rust and was named by JSON path (review 2026-09-26, F4).
    (
        po.spec.ewridge,
        dict(targets=[3]),
        "targets must be a list of strs, po.target tables or window expressions "
        "looking ahead, got list [3]",
    ),
    (
        po.spec.ewridge,
        dict(targets=[{"column": "p", "name": 1}]),
        "targets must be a list of strs, po.target tables or window expressions "
        "looking ahead, got list [{'column': 'p', 'name': 1}]",
    ),
    (
        po.spec.ewridge,
        dict(targets=[{"col": "p"}]),
        "targets must be a list of strs, po.target tables or window expressions "
        "looking ahead, got list [{'col': 'p'}]",
    ),
    (
        po.spec.ewridge,
        dict(targets=[None]),
        "targets must be a list of strs, po.target tables or window expressions "
        "looking ahead, got list [None]",
    ),
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
        po.spec.ew_cov("m", features=["x0", "y"], targets=["y"], half_life=10.0)


def test_kmeans_has_no_targets():
    with pytest.raises(TypeError, match='spec "m": kmeans\\(\\) takes no targets'):
        po.spec.kmeans("m", features=["x0", "y"], targets=["y"], k=2, half_life=10.0)


def test_ew_class_takes_a_label_not_targets():
    with pytest.raises(TypeError, match='spec "m": ew_class\\(\\) takes `label`, not targets'):
        po.spec.ew_class(
            "m",
            features=["x0"],
            label="y",
            targets=["y"],
            classes=["a", "b"],
            precision_prior=1.0,
            half_life=10.0,
        )


VALUES = [
    # A list's entries are held to the int floor as a scalar is (review
    # 2026-09-26, F7: a negative lag was named by serde as `model`).
    (po.spec.marginal, dict(lags=[-1]), "lags must be >= 1, got list [-1]"),
    (
        po.spec.marginal,
        dict(lags=[1], cross_lags=[-1]),
        "cross_lags must be >= 1, got list [-1]",
    ),
    # A clock parameter since task 178: the Rust side's message, in clock
    # units; its row cap is a count of at least one.
    (
        po.spec.ewridge,
        dict(coef_every=-1),
        "coef_every must be finite and >= 0 clock units (0 writes `coef` on every row), got -1",
    ),
    (
        po.spec.ewridge,
        dict(max_rows_between_coefs=0),
        "max_rows_between_coefs must be >= 1, got 0",
    ),
    # No sweep is no descent: every solve a failure, and coefficients that
    # read like a fit (review 2026-10-05, PB7).
    (po.spec.lasso, dict(lasso_path=[0.1], max_iter=-1), "max_iter must be >= 1, got -1"),
    (po.spec.lasso, dict(lasso_path=[0.1], max_iter=0), "max_iter must be >= 1, got 0"),
    (po.spec.ewridge, dict(solve_every=INF), "solve_every must be finite, got float inf"),
    (po.spec.ewridge, dict(resid_quantiles=[0.5, INF]), "resid_quantiles must be finite"),
    (po.spec.rls, dict(ridge=-INF), "ridge must be finite, got float -inf"),
    (po.spec.ewridge, dict(conformal=INF), "conformal must be finite, got float inf"),
    (
        po.spec.ewridge,
        dict(conformal=1.0),
        "conformal must be a coverage level strictly between 0 and 1, got 1",
    ),
    (
        po.spec.ewridge,
        dict(conformal_rate=0.1),
        "conformal_rate needs conformal (the coverage level) to be set",
    ),
    (
        po.spec.ewridge,
        dict(conformal=0.9, conformal_rate=0.0),
        "conformal_rate must be finite and > 0, got 0",
    ),
    (
        po.spec.ew_cov,
        dict(features=["x0", "y"], targets=None, stats=["mahal"]),
        "ew_cov mahal needs `precision_prior`",
    ),
    (
        po.spec.ew_cov,
        dict(features=["x0", "y"], targets=None, mahal_quantiles=[0.5]),
        'ew_cov mahal_quantiles needs "mahal" in `stats`',
    ),
    (
        po.spec.ew_cov,
        dict(features=["x0", "y"], targets=None, pca=3),
        "ew_cov pca asks for 3 components of 2 features",
    ),
    (
        po.spec.ew_cov,
        # Clock units since task 161, 0 for every row, as `solve_every`.
        dict(features=["x0", "y"], targets=None, pca=1, pca_every=-1.0),
        "ew_cov pca_every must be finite and >= 0 clock units (0 refreshes on every row), got -1",
    ),
    (
        po.spec.ew_cov,
        dict(features=["x0", "y"], targets=None, pca_every=2),
        "ew_cov pca_every needs `pca`",
    ),
    (
        po.spec.ew_cov,
        dict(features=["x0", "y"], targets=None, max_rows_between_pca=20),
        "ew_cov max_rows_between_pca needs `pca`",
    ),
    (
        po.spec.ew_class,
        dict(targets=None, label="y", classes=["a"], precision_prior=1.0),
        "ew_class classes must list at least 2 classes (got 1)",
    ),
    (
        po.spec.ew_class,
        dict(targets=None, label="y", classes=["a", "b", "a"], precision_prior=1.0),
        'ew_class classes lists "a" more than once',
    ),
    (
        po.spec.ew_class,
        dict(targets=None, label="y", classes=["a", ""], precision_prior=1.0),
        "ew_class classes must not contain an empty name",
    ),
    (
        po.spec.ew_class,
        dict(targets=None, label="y", classes=["a", "b"], precision_prior=1.0, covariance="lda"),
        'unknown ew_class covariance "lda" (expected full, shared or diagonal)',
    ),
    # `hmm`'s named `ew_class` (task 190's find, folded into task 193).
    (
        po.spec.hmm,
        dict(features=["x0", "y"], targets=None, k=2, precision_prior=0.1, covariance="lda"),
        'unknown hmm covariance "lda" (expected full, shared or diagonal)',
    ),
    (
        po.spec.ew_class,
        dict(targets=None, label="y", classes=["a", "b"], precision_prior=0.0),
        "ew_class precision_prior must be finite and > 0, got 0",
    ),
    # Every refusal of a value names it (review 2026-10-06, PC11): about
    # thirty named the parameter and not the value.
    (po.spec.ewridge, dict(ridge=-1.0), "ridge must be finite and >= 0, got -1"),
    (po.spec.ewridge, dict(min_weight=-1.0), "min_weight must be >= 0, got -1"),
    (
        po.spec.ewridge,
        dict(solve_every=-1.0),
        "solve_every must be finite and >= 0 (0 solves every row), got -1",
    ),
    (po.spec.ewridge, dict(half_life=-1.0), 'half_life must be > 0 ("inf" for no decay), got -1'),
    (po.spec.ewridge, dict(half_life=None, lam=1.5), "lam must be in (0, 1], got 1.5"),
    (
        po.spec.ewridge,
        dict(emit_drift=True, drift_delta=-1.0),
        "drift_delta must be finite and >= 0, got -1",
    ),
    (
        po.spec.ewridge,
        dict(emit_drift=True, drift_threshold=-1.0),
        "drift_threshold must be finite and > 0, got -1",
    ),
    (
        po.spec.ewridge,
        dict(emit_drift=True, drift_action="nope"),
        'drift_action must be "flag" or "reset", got "nope"',
    ),
    (
        po.spec.ewridge,
        dict(resid_quantiles=[1.5]),
        "resid_quantiles must be strictly between 0 and 1, got 1.5",
    ),
    (
        po.spec.ewridge,
        dict(ridge=[1e-6, 1.0], emit_averaged=True, average_eta=-1.0),
        'average_eta must be > 0 ("inf" is emit_selected\'s argmin), got -1',
    ),
    (
        po.spec.ewridge,
        dict(window_size=5.0, window_every=-1.0),
        "window_every must be finite and >= 0 clock units (0 snapshots every row), got -1",
    ),
    (
        po.spec.ewridge,
        dict(session="s", session_gap=0.0, session_shrink=2.0, long_half_life=5.0),
        "session_shrink must be in [0, 1], got 2",
    ),
    (
        po.spec.ewridge,
        dict(session="s", session_gap=0.0, session_shrink=0.5, long_half_life=-1.0),
        "long_half_life must be > 0, got -1",
    ),
    (po.spec.pa, dict(c=-1.0), 'pa c must be > 0 ("inf" caps nothing: mode "pa"), got -1'),
    (po.spec.pa, dict(eps=-1.0), "pa eps must be finite and >= 0, got -1"),
    (po.spec.sgd, dict(learning_rate=-1.0), "learning_rate must be finite and > 0, got -1"),
    (
        po.spec.sgd,
        dict(loss="huber", huber_delta=-1.0),
        'huber_delta must be > 0 ("inf" is the squared loss), got -1',
    ),
    (
        po.spec.huber,
        dict(huber_delta=-1.0),
        'huber_delta must be > 0 ("inf" is least squares), got -1',
    ),
    (po.spec.huber, dict(ridge=-1.0), "ridge must be finite and >= 0, got -1"),
    (po.spec.quantile, dict(quantile=1.5), "quantile must be in (0, 1), got 1.5"),
    (
        po.spec.quantile,
        dict(quantile=0.5, quantile_eps=-1.0),
        "quantile_eps must be finite and > 0, got -1",
    ),
    (
        po.spec.ew_cov,
        dict(features=["x0", "y"], targets=None, stats=["mahal"], precision_prior=-1.0),
        "precision_prior must be finite and > 0, got -1",
    ),
    (
        po.spec.holt,
        dict(features=None, half_life=None, level_half_life=-1.0),
        "level_half_life must be > 0, got -1",
    ),
    (
        po.spec.holt,
        dict(features=None, trend_half_life=-1.0),
        'trend_half_life must be > 0 ("inf" forgets no slope), got -1',
    ),
    (
        po.spec.lasso,
        dict(lasso_path=[0.1], select_half_life=-1.0),
        "select_half_life must be > 0, got -1",
    ),
    (po.spec.lasso, dict(lasso_path=[0.1], tol=-1.0), "tol must be finite and > 0, got -1"),
    (po.spec.lasso, dict(lasso_path=[0.1], l1_ratio=2.0), "l1_ratio must be in [0, 1], got 2"),
    (
        po.spec.lasso,
        dict(lasso_path=[0.1, 0.2]),
        "lasso_path must be strictly decreasing, got 0.1 then 0.2",
    ),
    (
        po.spec.lasso,
        dict(lasso_path=[0.1, -0.1]),
        "lasso_path values must be finite and >= 0, got -0.1",
    ),
    (
        po.spec.kalman,
        dict(coef_half_life=-1.0),
        'coef_half_life must be > 0 ("inf" pins a coefficient), got -1',
    ),
    (
        po.spec.kalman,
        dict(coef_half_life=10.0, revert_half_life=-1.0),
        'revert_half_life must be > 0 ("inf" is the random walk), got -1',
    ),
    (
        po.spec.kalman,
        dict(q=[-1.0, 0.0]),
        "q values must be finite and >= 0 (0 pins a coefficient), got -1",
    ),
    (po.spec.kalman, dict(q=[0.0]), "q must have length 2, got 1"),
    (
        po.spec.kalman,
        dict(coef_half_life=10.0, obs_var=-1.0),
        "obs_var must be finite and > 0, got -1",
    ),
    (po.spec.kalman, dict(coef_half_life=10.0, p0=-1.0), "p0 must be finite and > 0, got -1"),
    (po.spec.ftrl, dict(alpha=-1.0), "ftrl alpha must be finite and > 0, got -1"),
    (po.spec.ftrl, dict(beta=-1.0), "ftrl beta must be finite and >= 0, got -1"),
    (po.spec.ftrl, dict(l1=-1.0), "ftrl l1 must be finite and >= 0, got -1"),
    (po.spec.ftrl, dict(l2=-1.0), "ftrl l2 must be finite and >= 0, got -1"),
    (po.spec.rls, dict(ridge=-1.0), "rls ridge must be finite and > 0, got -1"),
    (
        po.spec.kmeans,
        dict(features=["x0", "y"], targets=None, k=2, split_merge=-1.0),
        "split_merge must be finite and >= 0 (0 disables it), got -1",
    ),
    (
        po.spec.kmeans,
        dict(features=["x0", "y"], targets=None, k=2, dead_frac=-1.0),
        "dead_frac must be finite and >= 0 (0 disables it), got -1",
    ),
    (
        po.spec.kmeans,
        dict(features=["x0", "y"], targets=None, k=2, scale_floor=-1.0),
        "scale_floor must be finite and >= 0 (0 is the EW variance alone), got -1",
    ),
    (
        po.spec.micro,
        dict(features=["x0", "y"], targets=None, eps=-1.0),
        "micro eps must be finite and > 0, got -1",
    ),
    (
        po.spec.micro,
        dict(features=["x0", "y"], targets=None, eps=0.3, beta_mu=-1.0),
        "beta_mu must be finite and > 0, got -1",
    ),
    (
        po.spec.micro,
        dict(features=["x0", "y"], targets=None, eps=0.3, macro_link=-1.0),
        "macro_link must be finite and >= 0 (0 links nothing), got -1",
    ),
    (
        po.spec.micro,
        dict(features=["x0", "y"], targets=None, eps=0.3, prune_every=-1.0),
        "micro prune_every must be finite and >= 0 clock units (0 checkpoints on every row), "
        "got -1",
    ),
]


@pytest.mark.parametrize("builder,kw,msg", VALUES, ids=[m.split(" must")[0] for _, _, m in VALUES])
def test_a_bad_value_names_the_parameter(builder, kw, msg):
    merged = {k: v for k, v in {**BASE, **kw}.items() if v is not None}
    with pytest.raises(ValueError) as exc:
        builder("m", **merged)
    assert msg in str(exc.value), str(exc.value)


def test_numpy_scalars_are_plain_numbers():
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0"], half_life=np.float64(10.0), coef_every=np.int64(3)
    )
    assert spec["half_life"] == 10.0 and spec["coef_every"] == 3
    out = po.ModelBank([spec]).fit_predict(_df())
    assert out["m"].struct.field("coef").null_count() < out.height


BUILDERS = {
    po.spec.ewridge: {},
    po.spec.rls: {},
    po.spec.lasso: dict(lasso_path=[0.1]),
    po.spec.kalman: dict(coef_half_life=10.0),
    po.spec.huber: {},
    po.spec.quantile: dict(quantile=0.5),
    po.spec.ftrl: {},
    po.spec.ew_cov: dict(features=["x0", "y"], targets=None),
    po.spec.sgd: {},
    po.spec.pa: {},
    po.spec.holt: dict(features=None),
    po.spec.kmeans: dict(features=["x0", "y"], targets=None, k=2),
    po.spec.ew_class: dict(targets=None, label="y", classes=["a", "b"], precision_prior=1.0),
    po.spec.marginal: {},
    # The seven the sweeps below never saw until task 111.
    po.spec.micro: dict(features=["x0", "y"], targets=None, eps=0.3),
    po.spec.seqtest: dict(features=None, half_life=None),
    po.spec.deco: dict(features=["x0", "y"], targets=None),
    po.spec.rcov: dict(
        features=["x0", "y"],
        targets=None,
        half_life=None,
        group="g",
        group_close="monotone",
        block_rows=100,
    ),
    po.spec.hmm: dict(features=["x0", "y"], targets=None, k=2, precision_prior=0.1),
    po.spec.corrchange: dict(features=["x0", "y"], targets=None, half_life=None, span_rows=20),
    po.spec.bocpd: dict(features=["x0", "y"], targets=None, half_life=None),
}


#: The models with no target, whose builders write ``features[0]`` as the
#: target the plumbing needs.
NO_TARGET = [
    po.spec.ew_cov,
    po.spec.kmeans,
    po.spec.micro,
    po.spec.deco,
    po.spec.bocpd,
    po.spec.corrchange,
    po.spec.hmm,
    po.spec.rcov,
]


@pytest.mark.parametrize("builder", NO_TARGET, ids=lambda b: b.__name__)
def test_an_empty_features_list_is_named_by_a_model_with_no_target(builder):
    """Review 2026-10-05 (YA1): these builders read ``features[0]`` before
    anything checked the list, so ``features=[]`` raised ``IndexError`` and the
    Rust side's own message was never reached; they say it first now."""
    merged = {k: v for k, v in {**BASE, **BUILDERS[builder]}.items() if v is not None}
    with pytest.raises(ValueError, match='^spec "m": features must be non-empty'):
        builder("m", **{**merged, "features": []})


@pytest.mark.parametrize(
    "builder",
    [po.spec.deco, po.spec.bocpd, po.spec.corrchange, po.spec.hmm, po.spec.rcov],
    ids=lambda b: b.__name__,
)
def test_a_model_with_no_target_refuses_targets_by_name(builder):
    """Review 2026-10-05 (YA5): five builders passed ``targets=`` on to
    ``_common`` beside their own, and the ``TypeError`` named ``_common()``
    with "multiple values"; they refuse it as ``ew_cov`` does."""
    merged = {k: v for k, v in {**BASE, **BUILDERS[builder]}.items() if v is not None}
    with pytest.raises(TypeError, match=rf'^spec "m": {builder.__name__}\(\) takes no targets'):
        builder("m", **merged, targets=["y"])


def test_the_float_sweeps_name_every_builder():
    """``BUILDERS`` named fourteen of the twenty-one builders, so the two
    sweeps below never tried an infinite or NaN float on the other seven
    (docs/PLAN.md task 111); it is held to the registry now."""
    from test_model_registry import MINIMAL

    assert {b.__name__ for b in BUILDERS} == set(MINIMAL)


def _members(hint) -> tuple:
    if typing.get_origin(hint) in (types.UnionType, typing.Union):
        return typing.get_args(hint)
    return (hint,)


def _inf_shaped_like(hint):
    """``"inf"`` in the shape the annotation asks for: bare for a float, nested
    once per ``list[...]`` otherwise (``q`` -> ``["inf"]``, ``coef_prior`` ->
    ``[["inf"]]``), and ``None`` for a list of anything else: ``stats``,
    ``classes``, ``lags`` and ``cross_lags`` were swept as ``[None]``, a
    refusal of the list's type that said nothing of ``inf`` (review
    2026-10-06, YA10)."""
    members = _members(hint)
    if float in members:
        return "inf"
    for m in members:
        if typing.get_origin(m) is list:
            inner = _inf_shaped_like(typing.get_args(m)[0])
            return None if inner is None else [inner]
    return None


def _float_parameters(builder) -> dict[str, typing.Any]:
    hints = typing.get_type_hints(builder.__wrapped__) | typing.get_type_hints(_spec._common)
    skip = {"name", "model", "return", "targets", "features"}
    out = {}
    for key, hint in hints.items():
        if key not in skip and (inf := _inf_shaped_like(hint)) is not None:
            out[key] = inf
    return out


#: What a parameter needs beside it to be refused for its value rather than
#: for a companion it lacks: a clock under ``gap_cap`` (with a finite cap to
#: build with), a window under ``window_every``.
COMPANIONS: dict[str, dict[str, typing.Any]] = {
    "gap_cap": {"clock": "t", "gap_cap": 1.0},
    "window_every": {"window_size": 10.0},
    # `q` is the process noise `coef_half_life` would derive, and `kalman`
    # takes exactly one of the two (review round 4, PC6): a finite `q`, one
    # entry per coefficient of `BASE`'s, stands in until the sweep's own.
    "q": {"coef_half_life": None, "q": [0.0, 0.0]},
}

#: A refusal that speaks of the value being infinite.
_SAYS_INF = re.compile(r"finite|\binf\b|infinit")


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_the_inf_table_matches_the_rust_side(builder):
    """``_INF_OK`` says which parameters may be infinite. Feed ``"inf"``
    straight to Rust for every float parameter: an allowed one must get past
    the parser *and* past ``validate``'s finiteness checks (a refusal for want
    of another parameter is fine), and a refused one must be refused by Rust
    too, for being infinite, or the Python check is inventing a rule. It
    checked the parser alone, which is the field's type, while ``validate``
    refused two of the table's entries (review 2026-09-12, S27).

    And it took any ``ValueError`` as that refusal: 39 of 330 pairs were
    refused for something else, ``gap_cap`` for want of a clock,
    ``window_every`` of a window, ``q`` for its length (review 2026-10-06,
    YA10). Each now has its companion and the right shape, and the refusal
    must speak of the value being infinite, or, for a parameter the kind
    does not take at all (``coef_every`` where there is no coefficient,
    ``embargo`` beside a closed group), name it."""
    allowed = _spec._INF_OK["*"] | _spec._INF_OK.get(builder.__name__, frozenset())
    kwargs = {k: v for k, v in {**BASE, **BUILDERS[builder]}.items() if v is not None}
    for key, inf in _float_parameters(builder).items():
        spec = builder("m", **{**kwargs, **COMPANIONS.get(key, {})})
        where = spec if key in spec else spec["model"]
        # A key the Rust spec skips when absent is written only when given.
        assert key in where or key in _spec._SKIPPED_WHEN_ABSENT.get(builder.__name__, ()), (
            f"{builder.__name__}.{key} is not a key of the spec dict"
        )
        if key == "q":
            # One entry per coefficient: the features and the intercept.
            inf = ["inf"] * (len(spec["features"]) + 1)
        where[key] = inf
        try:
            po.ModelBank([spec])
        except ValueError as e:
            msg = str(e)
            if key in allowed:
                assert "invalid spec" not in msg, (key, msg)
                assert "finite" not in msg, (key, msg)
            else:
                about_inf = _SAYS_INF.search(msg) is not None
                not_taken = key in msg and key not in COMPANIONS and key != "q"
                assert about_inf or not_taken, (builder.__name__, key, msg)
        else:
            assert key in allowed, f"{builder.__name__}.{key} accepts inf but is not in _INF_OK"


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_nan_is_no_setting_for_any_float_parameter(builder):
    """The NaN twin of the `inf` sweep above: there is no parameter for which
    NaN means something, so the bank must refuse the word ``"nan"`` -- which
    a hand-written JSON spec can carry, the Python builders refusing it before
    anything is serialized -- for every float parameter of every builder.
    Two doors: a key that accepts words (``"inf"``) parses ``"nan"`` too and
    must then be refused *by name* from a validator; a plain float key never
    parses the word, and the parser's type error is the refusal (JSON has no
    NaN to carry into it; TOML's ``nan`` literal is the Rust side's
    ``spec_inf.rs``). The spec layer named every key it validates itself; the
    core validators of ``sgd`` (``clip_gradient``, ``power``, ``l2``,
    ``eps``) and ``kalman`` (``half_life``, ``p0``, ``q``, ``obs_var``) tested
    ``v <= 0.0`` / ``v < 0.0``, which a NaN passes, and ``rls`` never checked
    its prior's entries: a NaN ``clip_gradient`` then panicked in
    ``f64::clamp`` on the first learned row, and a NaN ``obs_var`` made
    ``kalman`` predict its prior for the life of the stream with no error
    (review 2026-09-18, B4)."""
    kwargs = {k: v for k, v in {**BASE, **BUILDERS[builder]}.items() if v is not None}
    swept = 0
    for key, inf in _float_parameters(builder).items():
        if "inf" not in repr(inf):
            continue  # a list of names or lags: no float to make a NaN of
        spec = builder("m", **kwargs)
        where = spec if key in spec else spec["model"]
        # A key the Rust spec skips when absent is written only when given.
        assert key in where or key in _spec._SKIPPED_WHEN_ABSENT.get(builder.__name__, ()), (
            f"{builder.__name__}.{key} is not a key of the spec dict"
        )
        where[key] = _nan_shaped_like(inf)
        with pytest.raises(ValueError) as caught:
            po.ModelBank([spec])
        msg = str(caught.value)
        assert key in msg or 'string "nan", expected f64' in msg, (key, msg)
        swept += 1
    assert swept > 0, f"{builder.__name__} has no float parameter to sweep"


def _nan_shaped_like(inf):
    """``"inf"`` in whatever nesting `_inf_shaped_like` chose, as ``"nan"``."""
    if inf is None:
        return None
    return "nan" if inf == "inf" else [_nan_shaped_like(v) for v in inf]


# --- S27: inf where it means something, refused in both layers elsewhere ----

#: Where ``inf`` means something (review 2026-09-12, S27; the user's decision
#: of 2026-09-15): the builder takes it, and the bank builds with it. Each
#: entry carries what the parameter needs beside it to be valid at all.
INF_MEANS_SOMETHING = [
    (po.spec.huber, "huber_delta", {}),  # least squares
    (po.spec.sgd, "huber_delta", dict(loss="huber")),  # least squares
    (po.spec.ewridge, "long_half_life", dict(session="s", session_gap=1.0, session_shrink=0.5)),
    (po.spec.lasso, "select_half_life", {}),  # selection over the whole history
    (po.spec.holt, "level_half_life", dict(half_life=None)),  # the cumulative fit (S30)
    (po.spec.pa, "c", dict(mode="pa1")),  # mode "pa": the step is not capped
    (po.spec.ewridge, "average_eta", dict(ridge=[1e-6, 1.0], emit_averaged=True)),  # the argmin
]

#: Where it means nothing: the builder refuses it, and so does the bank when
#: the JSON carries ``"inf"``.
INF_MEANS_NOTHING = [
    (po.spec.ewridge, "drift_delta", dict(emit_drift=True), math.inf),
    (po.spec.ewridge, "drift_threshold", dict(emit_drift=True), math.inf),
    (po.spec.quantile, "quantile_eps", {}, math.inf),
    (po.spec.pa, "eps", {}, math.inf),
    (po.spec.ftrl, "alpha", {}, math.inf),
    (po.spec.ftrl, "beta", {}, math.inf),
    (po.spec.ftrl, "l1", {}, math.inf),
    (po.spec.ftrl, "l2", {}, math.inf),
    (po.spec.sgd, "learning_rate", {}, math.inf),
    (po.spec.ewridge, "ridge", {}, math.inf),
    # `q` alone: beside `coef_half_life` the pair is refused first (review
    # 2026-10-06, PC6).
    (po.spec.kalman, "q", dict(coef_half_life=None, q=[0.0, 0.5]), [math.inf, 0.5]),
]


def _kwargs(builder, extra):
    return {k: v for k, v in {**BASE, **BUILDERS[builder], **extra}.items() if v is not None}


@pytest.mark.parametrize(
    ("builder", "key", "extra"),
    INF_MEANS_SOMETHING,
    ids=[f"{b.__name__}.{k}" for b, k, _ in INF_MEANS_SOMETHING],
)
def test_inf_is_taken_where_it_means_something(builder, key, extra):
    """A half-life that forgets nothing, a Huber loss that is least squares, a
    step that is not capped, weights that are the argmin: the builder takes
    ``inf`` and the bank builds with it. Python refused every one, and so did
    Rust's parser, while its validation let TOML's ``inf`` through."""
    po.ModelBank([builder("m", **{**_kwargs(builder, extra), key: math.inf})])


@pytest.mark.parametrize(
    ("builder", "key", "extra", "value"),
    INF_MEANS_NOTHING,
    ids=[f"{b.__name__}.{k}" for b, k, _, _ in INF_MEANS_NOTHING],
)
def test_inf_is_refused_where_it_means_nothing(builder, key, extra, value):
    """A learning rate, a penalty, a tube or a threshold at ``inf`` is no
    setting, and both layers say so by name: the builder's own check, whose
    message ends ``got ...``, and the bank's, from the JSON. ``ridge`` and
    ``q`` were in the Python table and refused by Rust alone, which the
    builder reached anyway, since it validates through Rust."""
    kwargs = _kwargs(builder, extra)
    with pytest.raises(ValueError, match=f"{key} must be finite, got"):
        builder("m", **{**kwargs, key: value})
    spec = builder("m", **kwargs)
    where = spec if key in spec else spec["model"]
    where[key] = (
        [("inf" if math.isinf(v) else v) for v in value] if isinstance(value, list) else "inf"
    )
    with pytest.raises(ValueError):
        po.ModelBank([spec])


# --- hand-built dicts: serde names the path, and the visitors say what fits --


def test_a_hand_built_dict_is_checked_by_path():
    base = dict(name="m", model={"type": "ew_ridge"}, targets=["y"], features=["x0"])
    for bad, msg in [
        (dict(targets="y"), '[0].targets: invalid type: string "y", expected a sequence'),
        (dict(half_life="10"), '[0].half_life: "10" is not a duration: 10 has no unit'),
        (dict(half_life=[10, "x"]), '[0].half_life[1]: "x" is not a duration'),
        (
            dict(half_life=10, session_gap=[1]),
            "[0].session_gap: invalid type: sequence, expected a gap in clock units",
        ),
        (dict(half_life=10, model={"type": "ew_rdige"}), "[0].model.type: unknown variant"),
        (dict(half_life=10, model={}), "[0].model: missing field `type`"),
        (dict(half_life=10, targets=[3]), "[0].targets[0]: invalid type: integer `3`"),
        (dict(half_life=10, targets="y"), "[0].targets: invalid type: string"),
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
        ("target", dict(targets=[po.target("nope", relative_to="y")])),
        ("relative_to", dict(targets=[po.target("y", relative_to="nope")])),
        ("clock", dict(clock="nope", gap_cap=5.0)),
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
    # "input", not "frame": the same message serves the Arrow path, whose
    # caller has a chunk and no frame (review 2026-09-17, A5).
    assert 'the input has columns ["t", "x0", "y", "s"]' in text, text


@pytest.mark.parametrize(
    "role,kw",
    [
        ("feature", dict(features=["s"])),
        ("target", dict(targets=["s"])),
        ("relative_to", dict(targets=[po.target("y", relative_to="s")])),
        ("clock", dict(clock="s", gap_cap=5.0)),
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


#: Two instants an hour apart that Amsterdam shows at one wall time,
#: 02:30, once in summer time and once in winter time.
_FALL_BACK = [datetime.datetime(2024, 10, 27, 0, 30), datetime.datetime(2024, 10, 27, 1, 30)]


def zoned_keys(tz: str | None, n: int = 60) -> pl.DataFrame:
    """Two groups and two sessions keyed by instants: as naive UTC
    Datetimes with ``tz`` None, else the same instants shown in ``tz``."""
    utc = pl.Series([_FALL_BACK[i % 2] for i in range(n)], dtype=pl.Datetime("us"))
    day = pl.Series([_FALL_BACK[i * 2 // n] for i in range(n)], dtype=pl.Datetime("us"))
    if tz is not None:
        utc = utc.dt.replace_time_zone("UTC").dt.convert_time_zone(tz)
        day = day.dt.replace_time_zone("UTC").dt.convert_time_zone(tz)
    rng = np.random.default_rng(160)
    x = rng.standard_normal(n)
    return pl.DataFrame(
        {
            "g": utc,
            "s": day,
            "t": np.arange(n, dtype=float),
            "x0": x,
            "y": 2.0 * x + 0.1 * rng.standard_normal(n),
        }
    )


def test_a_zoned_datetime_group_is_keyed_by_its_instant():
    """Task 160, PA4b: a zoned Datetime group or session column was refused
    with polars' own message, this build formatting no time zone. A zoned
    key is its instant: the same numbers and keys as the UTC instants as a
    naive Datetime column, whatever zone shows it, two instants Amsterdam
    shows at one wall time are two groups, and every way a key goes back
    in -- ``drop_groups``, ``skip_learned`` -- takes it."""
    specs = [
        _spec_dict(group="g", clock="t", gap_cap=10.0),
        po.spec.ew_cov(
            "c", features=["x0", "y"], group="g", session="s", group_close="session", half_life=5.0
        ),
    ]
    naive = zoned_keys(None)
    want_bank = po.ModelBank(specs)
    want = want_bank.fit_predict(naive)
    want_closed = want_bank.closed_groups()
    assert want_bank.groups().filter(spec="m")["group"].to_list() == [
        "2024-10-27 00:30:00.000000",
        "2024-10-27 01:30:00.000000",
    ]
    for tz in ("Europe/Amsterdam", "America/New_York"):
        df = zoned_keys(tz)
        bank = po.ModelBank(specs)
        got = bank.fit_predict(df)
        assert got.select("m", "c").equals(want.select("m", "c")), tz
        assert bank.groups().equals(want_bank.groups()), tz
        assert bank.closed_groups().equals(want_closed), tz
        # A key as `groups` gives it goes back in.
        key = bank.groups()["group"][0]
        assert bank.drop_groups([key], spec="m") == 1, tz
        # So does the column itself, matched to the keys by `skip_learned`.
        half = po.ModelBank(specs)
        half.fit_predict(df.head(30))
        assert half.skip_learned(df).equals(df.slice(30)), tz


#: One column of every dtype polars can hand us, by name. Built lazily,
#: because a few need a cast to exist at all.
DTYPES: dict[str, typing.Callable[[], pl.Series]] = {
    "Int8": lambda: pl.Series("c", [1, 2], dtype=pl.Int8),
    "UInt8": lambda: pl.Series("c", [1, 2], dtype=pl.UInt8),
    "Int16": lambda: pl.Series("c", [1, 2], dtype=pl.Int16),
    "UInt16": lambda: pl.Series("c", [1, 2], dtype=pl.UInt16),
    "Int32": lambda: pl.Series("c", [1, 2], dtype=pl.Int32),
    "Int64": lambda: pl.Series("c", [1, 2], dtype=pl.Int64),
    "Int128": lambda: pl.Series("c", [1, 2], dtype=pl.Int128),
    "Float32": lambda: pl.Series("c", [1.0, 2.0], dtype=pl.Float32),
    "Decimal": lambda: pl.Series("c", [1.5, 2.5]).cast(pl.Decimal(10, 4)),
    "Boolean": lambda: pl.Series("c", [True, False]),
    "String": lambda: pl.Series("c", ["a", "b"]),
    "Categorical": lambda: pl.Series("c", ["a", "b"], dtype=pl.Categorical),
    "Enum": lambda: pl.Series("c", ["a", "b"], dtype=pl.Enum(["a", "b"])),
    "Binary": lambda: pl.Series("c", [b"a", b"b"]),
    "Date": lambda: pl.Series("c", [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)]),
    "Datetime": lambda: pl.Series(
        "c", [datetime.datetime(2026, 1, 1), datetime.datetime(2026, 1, 2)]
    ),
    "Time": lambda: pl.Series("c", [datetime.time(1), datetime.time(2)]),
    "Duration": lambda: pl.Series(
        "c", [datetime.timedelta(seconds=1), datetime.timedelta(seconds=2)]
    ),
    "List": lambda: pl.Series("c", [[1.0], [2.0]]),
    "Array": lambda: pl.Series("c", [[1.0, 2.0], [3.0, 4.0]], dtype=pl.Array(pl.Float64, 2)),
    "Struct": lambda: pl.Series("c", [{"a": 1.0}, {"a": 2.0}]),
    "Null": lambda: pl.Series("c", [None, None], dtype=pl.Null),
    "Object": lambda: pl.Series("c", [object(), object()], dtype=pl.Object),
}

#: Dtypes that are numbers once cast, and so are legal features.
NUMERIC = {
    "Int8",
    "UInt8",
    "Int16",
    "UInt16",
    "Int32",
    "Int64",
    "Int128",
    "Float32",
    "Decimal",
    "Boolean",
    "Null",
}


def _two_rows() -> pl.DataFrame:
    return pl.DataFrame({"y": [1.0, 2.0], "x0": [0.5, 1.5]})


@pytest.mark.parametrize("name", list(DTYPES), ids=list(DTYPES))
def test_a_column_the_spec_never_names_is_ignored_whatever_its_dtype(name):
    """A frame carries columns the model does not use, and the whole frame
    crosses into Rust. A `Decimal` or `Int128` column *anywhere* in it used to
    abort the process with `activate 'dtype-decimal' feature` -- a panic
    inside the conversion, before any validation of ours could name the
    column, on a column the spec never asked for. The dtype features are on
    now (IMPROVEMENTS U5); this table is what keeps them on."""
    df = _two_rows().with_columns(DTYPES[name]().alias("c"))
    spec = po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=5.0, min_weight=1.0)
    out = po.ModelBank([spec]).fit_predict(df)
    assert out["m"].struct.field("weight_sum").len() == 2


@pytest.mark.parametrize("name", list(DTYPES), ids=list(DTYPES))
def test_a_feature_of_any_dtype_is_either_used_or_named(name):
    """The other half: used as a feature, a dtype either casts to f64 or is
    refused by name. Never a panic, and never silently all-null."""
    df = _two_rows().with_columns(DTYPES[name]().alias("c"))
    spec = po.spec.ewridge("m", targets=["y"], features=["c"], half_life=5.0, min_weight=1.0)
    if name in NUMERIC:
        po.ModelBank([spec]).fit_predict(df)
        return
    with pytest.raises(ValueError) as exc:
        po.ModelBank([spec]).fit_predict(df)
    assert 'feature column "c"' in str(exc.value), str(exc.value)
    assert "must be numeric" in str(exc.value), str(exc.value)


@pytest.mark.parametrize("dtype", [pl.Int8, pl.UInt16, pl.Int128, pl.Float32, pl.Decimal(12, 4)])
def test_a_narrow_dtype_gives_the_same_answer_as_float64(dtype):
    """Casting happens on our side, so a `UInt8` or `Decimal` feature must fit
    exactly what the same numbers as `Float64` would."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    df = pl.DataFrame({"y": [2 * v + 1 for v in x], "x0": x})
    spec = po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=5.0, min_weight=2.0)
    want = po.ModelBank([spec]).fit_predict(df)
    got = po.ModelBank([spec]).fit_predict(df.with_columns(pl.col("x0").cast(dtype)))
    assert want.equals(got, null_equal=True)


def test_a_decimal_parquet_runs_through_the_cli(tmp_path, online_cli):
    """Prices in parquet are commonly ``Decimal`` and small codes ``UInt8``. A
    ``Decimal`` once aborted the process at the boundary (IMPROVEMENTS U5),
    which for the CLI meant a file it could not open at all; its test went
    with ``po.run`` in task 83, and the ``ModelBank`` test above never reached
    the CLI's parquet reader (docs/PLAN.md task 112). The CLI's output is the
    bank's on the same numbers as ``Float64``, to the bit."""
    n = 400
    rng = np.random.default_rng(83)
    x = rng.normal(size=n)
    src, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    df = pl.DataFrame(
        {
            "price": pl.Series(np.round(100.0 + x, 4)).cast(pl.Decimal(12, 4)),
            "code": pl.Series(np.arange(n) % 3).cast(pl.UInt8),
            "y": 2.0 * x + 0.1 * rng.normal(size=n),
        }
    )
    df.write_parquet(src)
    spec = po.spec.ewridge("m", targets=["y"], features=["price", "code"], half_life=50.0)
    run_online(online_cli, tmp_path, [spec], input=src, output=out)
    plain = df.with_columns(pl.col("price", "code").cast(pl.Float64))
    want = po.ModelBank([spec]).fit_predict(plain)["m"]
    got = pl.read_parquet(out)["m"]
    assert got.struct.field("pred_y").drop_nulls().len() > n // 2
    assert got.equals(want, null_equal=True)


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


@pytest.mark.parametrize("method", ["fit_predict", "fit_predict_arrow"])
def test_concurrent_fit_predict_says_so(method):
    """The GIL is released for the run, so a second thread *can* reach the
    bank; it is refused with a sentence, not pyo3's "Already borrowed" -- and
    the sentence names the method the caller used, the Arrow twin included
    (review 2026-09-17, T6)."""
    n = 200_000
    df = pl.DataFrame(
        {
            "x0": np.random.default_rng(0).standard_normal(n),
            "y": np.random.default_rng(1).standard_normal(n),
        }
    )
    bank = po.ModelBank([_spec_dict(half_life=[10.0, 100.0, 1000.0], coef_every=1)])
    call = getattr(bank, method)

    def go(start: threading.Barrier, errors: list, done: list) -> None:
        start.wait()
        try:
            call(df)
            done.append(1)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    # The overlap is made certain rather than hoped for: four threads leave a
    # barrier together on 200,000 rows each, and if one still finishes before
    # the others reach the bank, the round is run again; five rounds without
    # an overlap fail the test, which then tested nothing (review 2026-10-05,
    # TC10). A fast machine once turned it red.
    for _ in range(5):
        start, errors, done = threading.Barrier(4), [], []
        threads = [threading.Thread(target=go, args=(start, errors, done)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert done, errors
        if errors:
            break
    else:
        pytest.fail("four threads never overlapped in five rounds")
    for e in errors:
        assert isinstance(e, RuntimeError), e
        assert f"ModelBank.{method}:" in str(e), str(e)
        assert "in use on another thread" in str(e), str(e)
        assert "one ordered stream" in str(e)
    # The bank is intact: the winners' rows are counted, nothing else happened.
    assert bank.fit_predict(df.head(10)).height == 10


def _overlapping_calls(make, call) -> list[RuntimeError]:
    """Four threads leave a barrier together and `call` one object `make`
    built, a round at a time, until two calls overlap: the second finds the
    object in use and is refused with a `RuntimeError`. A call that holds
    the GIL for its whole run never overlaps another, so five rounds without
    one fail."""
    for _ in range(5):
        obj = make()
        start, errors = threading.Barrier(4), []

        def go(obj=obj, start=start, errors=errors) -> None:
            start.wait()
            try:
                call(obj)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=go) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        busy = [err for err in errors if isinstance(err, RuntimeError)]
        if busy:
            return busy
    pytest.fail("four threads never overlapped in five rounds: the call holds the GIL")


@pytest.mark.parametrize("what", ["RefreshTime.feed", "Windows.feed", "Windows.finish"])
def test_the_stream_operators_release_the_gil(what):
    """`refresh_time`'s and `with_windows`' native calls release the GIL while
    they work, as `fit_predict` does, so a 100,000-row chunk of an IO-plugin
    source does not stop every other Python thread for its length (review
    round 4, SF12). Proved by overlap: two threads inside one call at once,
    the second refused with a sentence naming the class and the method."""
    import json

    from polars_online import _polars_online as native

    n = 400_000
    big = pl.DataFrame(
        {
            "series": ["a", "b"] * (n // 2),
            "t": np.arange(n, dtype=float),
            "v": np.random.default_rng(0).standard_normal(n),
            "x": np.random.default_rng(1).standard_normal(n),
        }
    )
    # A forward window longer than the frame: every row waits for `finish`.
    tree = ["rewm_sum", ["col", "x"], {"half_life": 50.0, "window_size": 1e9}]
    config = json.dumps({"formulas": [{"name": "f", "tree": tree}]})

    def windows():
        return native.Windows(config, big.clear())

    def fed():
        w = windows()
        w.feed(big)
        return w

    cls, method = what.split(".")
    make, call = {
        "RefreshTime.feed": (
            lambda: native.RefreshTime(["a", "b"], "series", "t", "v"),
            lambda o: o.feed(big),
        ),
        "Windows.feed": (windows, lambda o: o.feed(big)),
        "Windows.finish": (fed, lambda o: o.finish()),
    }[what]
    for e in _overlapping_calls(make, call):
        assert f"{cls}.{method}: " in str(e), str(e)
        assert "in use on another thread" in str(e), str(e)


def test_ridge_scale_is_checked_by_name():
    """R2-P10: a wrong ``ridge_scale`` was refused by serde as an unknown
    variant with no parameter name."""
    with pytest.raises(ValueError, match="ridge_scale must be"):
        po.spec.ewridge("m", targets=["y"], features=["x"], ridge_scale="foo")


def test_gap_cap_without_a_clock_is_refused_by_name():
    """Task 160, PB2: ``gap_cap`` caps the clock's step, and without a clock
    each row is one step. It was taken and capped the row count: 0.5 halved
    every decay step, marked every row capped and cleared every lag, so an
    AR(0.9) input read ``lagcorr`` 0.0. Refused as ``restart_after_step_back``
    is, in a spec and in ``with_windows``."""
    with pytest.raises(ValueError, match='spec "m": gap_cap needs clock'):
        po.spec.ewridge("m", **BASE, gap_cap=0.5)
    with pytest.raises(ValueError, match='spec "c": gap_cap needs clock'):
        po.spec.ew_cov("c", features=["x0", "y"], half_life=10.0, lags=[1], gap_cap=0.5)
    with pytest.raises(ValueError, match="with_windows: gap_cap needs clock"):
        po.stream.with_windows(
            _df(), f=po.rewm_sum("x0", half_life=10.0, window_size=3.0), gap_cap=0.5
        )
    # With the clock, the same cap is a cap.
    po.spec.ewridge("m", **BASE, clock="t", gap_cap=0.5)


EMPTY_HALF_LIFE = 'spec "m": half_life names no half-life; give one or a grid'


@pytest.mark.parametrize(
    ("builder", "kw", "msg"),
    [
        (po.spec.ewridge, dict(half_life=[]), EMPTY_HALF_LIFE),
        (po.spec.rls, dict(half_life=[]), EMPTY_HALF_LIFE),
        (po.spec.sgd, dict(half_life=[]), EMPTY_HALF_LIFE),
        # The other lists that make a grid were refused already, each by name.
        (po.spec.ewridge, dict(ridge=[]), "ridge grid must have at least one value"),
        (po.spec.lasso, dict(lasso_path=[]), 'spec "m": lasso_path must be non-empty'),
    ],
)
def test_an_empty_grid_is_refused_by_name(builder, kw, msg):
    """Task 160, PB3: ``half_life = []`` passed the builder and the Rust
    validation and built no model instance at all, an output struct with no
    field. A grid of no values names no model."""
    with pytest.raises(ValueError, match=msg):
        builder("m", **{**BASE, **kw})


def test_an_empty_half_life_in_a_dict_spec_is_refused():
    """The same through the door JSON and TOML come in by. ``lam`` is one
    number, never a grid, so an empty list is no ``lam`` at either door."""
    spec = po.spec.ewridge("m", **BASE)
    spec["half_life"] = []
    with pytest.raises(ValueError, match=EMPTY_HALF_LIFE):
        po.ModelBank([spec])
    with pytest.raises(TypeError, match='spec "m": lam must be a number, got list'):
        po.spec.ewridge("m", targets=["y"], features=["x0"], lam=[])
    spec = po.spec.ewridge("m", targets=["y"], features=["x0"], lam=0.99)
    spec["lam"] = []
    with pytest.raises(ValueError, match="lam: invalid type: sequence, expected f64"):
        po.ModelBank([spec])


def test_a_deco_block_refusal_reads_as_one_sentence():
    """Task 160, PB8: two refusals were string literals continued across a
    line without a ``\\``, so each carried a run of 18 spaces."""
    spec = po.spec.deco("d", features=["x0", "y"], half_life=10.0)
    for blocks, want in [
        ([], "deco blocks is empty; leave it out for the unblocked equicorrelation, or name at"),
        (
            [["u", ["x0"]], ["u", ["y"]]],
            'deco block "u" is named twice; block names are the output labels, so they have',
        ),
    ]:
        spec["model"]["blocks"] = blocks
        with pytest.raises(ValueError) as exc:
            po.ModelBank([spec])
        assert want in str(exc.value), str(exc.value)
        assert "  " not in str(exc.value), str(exc.value)


# --- review round 4 (2026-10-06): ceilings, the spec's name, and the key -----


def _int_parameters(builder) -> dict[str, bool]:
    """Each int parameter of a builder, and whether it takes a list of them."""
    hints = typing.get_type_hints(builder.__wrapped__) | typing.get_type_hints(_spec._common)
    out = {}
    for key, hint in hints.items():
        members = _members(hint)
        if int in members:
            out[key] = False
        elif list[int] in members:
            out[key] = True
    return out


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_an_int_past_the_rust_width_is_refused_by_name(builder):
    """A count past what the Rust side holds it in was named by serde as
    ``model`` -- ``seed=2**64`` read ``model: invalid type: floating point``
    -- and the builder now refuses it by name with the value (review
    2026-10-06, YA4). The builder's table of widths is held to the Rust
    side: fed ``2**32`` straight from the dict, a ``u32`` field is refused in
    serde's words and no other is."""
    kwargs = {k: v for k, v in {**BASE, **BUILDERS[builder]}.items() if v is not None}
    for key, is_list in _int_parameters(builder).items():
        ceiling = _spec._int_ceiling(key)
        past = [ceiling + 1] if is_list else ceiling + 1
        with pytest.raises(ValueError, match=rf'spec "m": {key} must be <= {ceiling}, got'):
            builder("m", **{**kwargs, key: past})
        spec = builder("m", **kwargs)
        where = spec if key in spec else spec["model"]
        where[key] = [2**32] if is_list else 2**32
        try:
            po.ModelBank([spec])
        except ValueError as e:
            assert ("expected u32" in str(e)) == (key in _spec._U32), (key, str(e))
        else:
            assert key not in _spec._U32, f"{builder.__name__}.{key} took 2**32"


def test_a_count_past_its_width_is_named_in_python():
    with pytest.raises(ValueError, match=r'spec "m": seed must be <= 18446744073709551615, got 1'):
        po.spec.kmeans("m", features=["x0", "y"], k=2, seed=2**64, half_life=10.0)
    with pytest.raises(
        ValueError, match=r'spec "m": max_rows_between_solves must be <= 4294967295, got 4294967296'
    ):
        po.spec.ewridge("m", max_rows_between_solves=2**32, **BASE)


@pytest.mark.parametrize(
    ("builder", "kw", "msg"),
    [
        (po.spec.marginal, dict(lags=[1, 2**62]), "lags must be <= 1048576, got list"),
        (
            po.spec.ew_cov,
            dict(features=["x0", "y"], targets=None, lags=[2**40]),
            "lags must be <= 1048576, got list",
        ),
        (
            po.spec.corrchange,
            dict(
                features=["x0", "y"],
                targets=None,
                half_life=None,
                kind="window",
                span_rows=10,
                n_perm=2**21,
            ),
            "n_perm must be <= 1048576, got 2097152",
        ),
        (
            po.spec.ewridge,
            dict(emit_autocorr=True, resid_autocorr_lag=2**21),
            "resid_autocorr_lag must be <= 1048576, got 2097152",
        ),
    ],
    ids=["marginal.lags", "ew_cov.lags", "corrchange.n_perm", "resid_autocorr_lag"],
)
def test_a_count_that_sizes_memory_has_a_ceiling(builder, kw, msg):
    """A lag sizes a ring before the first row, and ``n_perm`` the draws held
    for a quantile: ``marginal(lags=[2**62])`` raised ``PanicException:
    capacity overflow`` inside the builder, and ``lags=[10**11]`` reserved 2.4
    TB (review 2026-10-06, CD10). The builder says so; so does the Rust side,
    from a dict, in its own words."""
    merged = {k: v for k, v in {**BASE, **kw}.items() if v is not None}
    with pytest.raises(ValueError, match=re.escape(f'spec "m": {msg}')):
        builder("m", **merged)
    legal = {"lags": [1], "n_perm": 200, "resid_autocorr_lag": 1}
    key = next(k for k in legal if k in kw)
    spec = builder("m", **{**merged, key: legal[key]})
    where = spec if key in spec else spec["model"]
    where[key] = kw[key]
    with pytest.raises(ValueError, match=re.escape(f"{key} must be at most 1048576 (2^20)")):
        po.ModelBank([spec])


def test_kmeans_k_and_buffer_have_ceilings_and_micro_needs_none():
    """``kmeans(k=10**9)`` and ``warm_rows=10**9`` were accepted and held
    that many rows before seeding; micro's linkage held ``max_clusters^2``
    doubles at every checkpoint (review 2026-10-06, CF2). ``k`` has a
    ceiling and the buffer a budget; micro's step is ``O(m)`` memory now, and
    its cap needs none."""
    kw = dict(features=["x0", "y"], half_life=10.0)
    with pytest.raises(ValueError, match=r'spec "m": kmeans: k must be at most 65536 \(2\^16\)'):
        po.spec.kmeans("m", k=2**16 + 1, warm_rows=2**16 + 1, **kw)
    with pytest.raises(ValueError, match=r"the warm-up buffer would hold 256\.00003 MiB"):
        po.spec.kmeans("m", k=3, warm_rows=5_592_406, **kw)
    po.spec.micro("m", eps=0.3, max_clusters=10**7, **kw)


def test_rcov_lagged_products_have_a_byte_ceiling():
    """``(max_bandwidth + 1) * k^2`` doubles before the first row: 84 GB at
    a bandwidth of 2^20 over 100 features, under the count ceiling (review
    2026-10-06, CE9)."""
    with pytest.raises(ValueError, match=r'spec "m": rcov: the lagged products would take 800 MiB'):
        po.spec.rcov(
            "m",
            features=[f"x{i}" for i in range(10)],
            group="g",
            group_close="monotone",
            bandwidth=2**20,
        )


def test_the_bin_budget_is_said_in_mib():
    """ "would need 0.00 GiB (...), over the 0.000001 MiB" (review 2026-10-06,
    CD16): one unit, and digits enough to read above the budget."""
    with pytest.raises(ValueError) as exc:
        po.spec.marginal("m", bins=4, bin_warm_rows=8, bin_budget=1e-6, **BASE)
    msg = str(exc.value)
    assert "GiB" not in msg and "0.00 " not in msg, msg
    assert re.search(r"would need 0\.000\d+ MiB .*, over the 0\.000001 MiB", msg), msg


def test_a_models_refusal_names_the_spec():
    """A model's own check named the model, not the spec: in a bank of several
    ``sgd: clip_gradient must be > 0`` did not say which (review 2026-10-06,
    YA4). And ``window_size must be > 0 (got inf)`` was untrue of infinity."""
    with pytest.raises(ValueError) as exc:
        po.spec.sgd("m", clip_gradient=0.0, **BASE)
    assert str(exc.value).startswith('spec "m": sgd: clip_gradient must be > 0'), str(exc.value)
    for model in (
        {"type": "lasso", "lasso_path": [0.1], "window_size": "inf"},
        {"type": "marginal", "window_size": "inf"},
    ):
        spec = {"name": "m", "model": model, "targets": ["y"], "features": ["x0"], "half_life": 10}
        with pytest.raises(ValueError) as exc:
            po.ModelBank([spec])
        assert str(exc.value) == 'spec "m": window_size must be finite and > 0 (got inf)'


def test_a_type_error_inside_the_model_names_the_key():
    """serde reads the model, an internally tagged enum, into a buffer first,
    so its path stopped at ``model``: ``[0].model: invalid type: string
    "inf", expected f64`` named no key (review 2026-10-06, PC11)."""
    for model, key in [
        ({"type": "micro", "eps": "inf"}, "eps"),
        ({"type": "sgd", "coef_sum": "inf"}, "coef_sum"),
    ]:
        spec = {"name": "m", "model": model, "targets": ["y"], "features": ["x0"], "half_life": 10}
        with pytest.raises(ValueError) as exc:
            po.ModelBank([spec])
        assert str(exc.value) == (
            f'invalid spec: [0].model.{key}: invalid type: string "inf", expected f64'
        )


def test_a_renamed_key_cites_nothing_a_wheel_lacks():
    """The rename refusal cited "docs/PLAN.md task 144", a file a wheel's user
    does not have (review 2026-10-06, PC11)."""
    spec = {"name": "m", "model": {"type": "ew_ridge"}, "targets": ["y"], "features": ["x0"]}
    with pytest.raises(ValueError) as exc:
        po.ModelBank([{**spec, "halflife": 10.0}])
    assert str(exc.value).endswith("; halflife was renamed half_life"), str(exc.value)
    assert "PLAN" not in str(exc.value)


def test_a_marginal_and_a_nameless_refusal_name_the_spec_as_every_other():
    """``marginal 'm':`` where every other refusal reads ``spec "m":``, and
    ``spec null:`` for a hand-built dict with no name (review 2026-10-06,
    YA6)."""
    with pytest.raises(ValueError, match=r'^spec "m": bin_edges is missing'):
        po.spec.marginal("m", bin_edges={"zz": [0.0]}, **BASE)
    with pytest.raises(ValueError, match=r'^spec "m": shards must be a number of shards'):
        po.spec.marginal("m", shards="many", **BASE)
    nameless = {"model": {"type": "ew_ridge"}, "targets": ["y"], "features": ["x0"]}
    with pytest.raises(ValueError, match="^a spec with no name: half_life must not be NaN$"):
        po.ModelBank([{**nameless, "half_life": math.nan}])


def test_a_clock_step_past_the_largest_double_is_refused_by_row():
    """``1e308 - (-1e308)`` overflows to infinity, and left the decayed
    clock's removed time infinite for good: every later stamp compared equal
    to any span, and ``coef_every`` wrote ``coef`` on every row (review
    2026-10-06, PB7). Refused by row, as a clock value that is not a number
    is, and the bank is left as it was."""
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0"], half_life=10.0, clock="t", gap_cap=10.0
    )
    frame = pl.DataFrame({"t": [-1e308, 1e308], "x0": [1.0, 2.0], "y": [1.0, 2.0]})
    bank = po.ModelBank([spec])
    with pytest.raises(ValueError) as exc:
        bank.fit_predict(frame)
    assert 'clock column "t" steps from -1e308 to 1e308 at row 1' in str(exc.value)
    assert bank.rows_seen() == 0


def test_a_mismatched_spec_on_load_is_named():
    """ "saved specs do not match the bank's specs" named neither the spec nor
    what differs (review 2026-10-06, PA6)."""
    specs = [
        po.spec.ewridge(n, targets=["y"], features=["x0"], half_life=h)
        for n, h in [("a", 10.0), ("b", 20.0), ("c", 30.0)]
    ]
    bank = po.ModelBank(specs)
    bank.fit_predict(_df())
    other = [specs[0], po.spec.ewridge("b", targets=["y"], features=["x0"], half_life=25.0)]
    with pytest.raises(ValueError, match="the file holds 3 specs and the bank 2"):
        po.ModelBank.load_bytes(bank.save_bytes(), other)
    other.append(specs[2])
    with pytest.raises(ValueError) as exc:
        po.ModelBank.load_bytes(bank.save_bytes(), other)
    assert str(exc.value) == (
        'saved specs do not match the bank\'s specs: spec "b" (2 of 3) differs at half_life: '
        "20.0 in the file, 25.0 in the bank; refusing to load"
    )
