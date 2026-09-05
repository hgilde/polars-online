"""The keyword parameters of every spec, as ``TypedDict`` classes (PEP 692).

The builders in ``_spec.py`` take the shared parameters as ``**common`` and the
expression namespace in ``_expr.py`` takes everything as ``**kwargs``; with
``**kwargs: Any`` an editor shows nothing and a typo is found at runtime.
Annotating them as ``Unpack[...]`` of the classes below gives completion and
type checking without changing a call (docs/IMPROVEMENTS.md U4).

Each class is a copy of a builder's signature, and a copy drifts, so
``tests/test_kwargs_typing.py`` pins every one to the builder it mirrors: same
keys, same annotations, same required set. Change the builder and the test
says which class to update. Defaults are not repeated here -- a TypedDict has
none; the builder's signature is where they live.

No ``from __future__ import annotations`` here: under it ``Required[...]`` is
a string the TypedDict machinery does not look inside, so every key would be
optional at runtime and the test below could not see the required ones.
"""

from typing import Required, TypedDict

__all__ = [
    "CommonKwargs",
    "EwCovKwargs",
    "EwridgeKwargs",
    "ExprKwargs",
    "FtrlKwargs",
    "HoltKwargs",
    "HuberKwargs",
    "KMeansKwargs",
    "MicroKwargs",
    "KalmanKwargs",
    "LassoKwargs",
    "PaKwargs",
    "QuantileKwargs",
    "RlsKwargs",
    "SgdKwargs",
]


class ExprKwargs(TypedDict, total=False):
    """The parameters every model shares, as the expression namespace takes
    them: ``_common``'s keywords minus what the expression itself supplies
    -- the target is the calling column, the features are the method's own
    argument, and grouping is ``.over(group)``."""

    add_intercept: bool
    clock: str | None
    halflife: float | list[float] | None
    lam: float | None
    max_dclock: float | None
    on_clock_reset: str
    session: str | None
    session_gap: float | str | None
    weight: str | None
    min_periods: float | list[float] | None
    coef_every: int
    emit_sigma: bool
    emit_resid_z: bool
    emit_selected: bool
    emit_averaged: bool
    average_eta: float | None
    emit_metrics: bool
    conformal: float | None
    conformal_rate: float | None
    resid_quantiles: list[float] | None
    emit_autocorr: bool
    resid_autocorr_lag: int | None
    emit_drift: bool
    drift_delta: float | None
    drift_threshold: float | None
    drift_action: str


class CommonKwargs(ExprKwargs, total=False):
    """What the builders take as ``**common``: the above plus the group."""

    group: str | None


# --- one per model: the builder's own parameters, over the shared ones -------


class EwridgeKwargs(ExprKwargs, total=False):
    ridge: float | list[float] | None
    feature_sets: dict[str, list[str]] | None
    standardize: bool
    ridge_decay: bool
    coef0: list[list[float]] | None
    session_shrink: float | None
    long_halflife: float | None
    solve_every: float | None
    max_rows_between_solves: int | None


class RlsKwargs(ExprKwargs, total=False):
    ridge: float | None
    coef0: list[list[float]] | None


class LassoKwargs(ExprKwargs, total=False):
    lasso_path: Required[list[float]]
    l1_ratio: float | None
    select_halflife: float | None
    solve_every: float | None
    max_rows_between_solves: int | None
    max_cd_iters: int | None
    cd_tol: float | None


class KalmanKwargs(ExprKwargs, total=False):
    coef_halflife: Required[float | list[float]]
    q: list[float] | None
    obs_var: float | None
    p0: float | None
    share_p: bool
    standardize: bool


class HuberKwargs(ExprKwargs, total=False):
    huber_delta: float | None
    ridge: float | None
    standardize: bool
    solve_every: float | None
    max_rows_between_solves: int | None


class QuantileKwargs(ExprKwargs, total=False):
    quantile: Required[float]
    ridge: float | None
    standardize: bool
    solve_every: float | None
    max_rows_between_solves: int | None
    quantile_eps: float | None


class FtrlKwargs(ExprKwargs, total=False):
    alpha: float | None
    beta: float | None
    l1: float | None
    l2: float | None
    strict_binary: bool
    loss: str


class EwCovKwargs(ExprKwargs, total=False):
    stats: list[str] | None
    precision_prior: float | None
    mahal_quantiles: list[float] | None
    pca: int | None
    pca_every: int | None


class SgdKwargs(ExprKwargs, total=False):
    loss: str
    huber_delta: float | None
    quantile: float | None
    eps: float | None
    learning_rate: float | None
    schedule: str
    power: float | None
    l2: float | None
    clip_gradient: float | None
    scale_features: bool


class PaKwargs(ExprKwargs, total=False):
    mode: str
    c: float | None
    eps: float | None


class HoltKwargs(ExprKwargs, total=False):
    level_halflife: float | None
    trend_halflife: float | None


class KMeansKwargs(ExprKwargs, total=False):
    k: Required[int]
    warm_rows: int | None
    seed_rule: str | None
    seed: int | None
    update_every: int | None
    split_merge: float | None
    sm_every: int | None
    dead_frac: float | None
    standardize: bool | None


class MicroKwargs(ExprKwargs, total=False):
    eps: Required[float]
    beta_mu: float | None
    max_clusters: int | None
    prune_every: int | None
    macro_link: float | None
    standardize: bool | None
