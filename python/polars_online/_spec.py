"""Spec builders: plain dicts, validated eagerly by the Rust core.

The public module is :mod:`polars_online.spec`; its docstring says what a
spec is, what every model shares and what a builder raises. Here: the
checker each builder is wrapped in (:func:`_checked`, a ``TypeError`` per
keyword whose value has the wrong shape, a ``ValueError`` for a count below
0 or a non-finite value where infinity means nothing), the shared
parameters (:func:`_common`, which has the Rust side validate the finished
dict), and the JSON the two sides exchange (:func:`_json`, where
``inf`` becomes the string the Rust side reads).
"""

from __future__ import annotations

import functools
import json
import math
import numbers
import types
import typing
from collections.abc import Callable
from typing import Any, Unpack

import polars as pl

from polars_online._kwargs import CommonKwargs
from polars_online._polars_online import (
    spec_coef_fields,
    spec_output_fields,
    spec_output_index,
    validate_spec,
)


def _json(spec: dict[str, Any] | list[dict[str, Any]]) -> str:
    """JSON has no infinity literal, but ``halflife=inf`` is meaningful (it pins
    a coefficient), so infinities are encoded as strings the Rust side
    understands. A NaN is never meaningful in a spec and is refused here, by
    parameter name, rather than by the JSON offset serde would report. NumPy
    scalars are plain numbers here (``json`` alone refuses them)."""

    def enc(v: Any, key: str) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, numbers.Integral):
            return int(v)
        if isinstance(v, numbers.Real):
            v = float(v)
            if math.isnan(v):
                who = spec.get("name") if isinstance(spec, dict) else None
                raise ValueError(f"spec {json.dumps(who)}: {key} must not be NaN")
            if math.isinf(v):
                return "inf" if v > 0 else "-inf"
            return v
        if isinstance(v, dict):
            return {k: enc(x, k) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [enc(x, key) for x in v]
        return v

    return json.dumps(enc(spec, "spec"))


def _from_json(text: str) -> Any:
    """The inverse of :func:`_json`: the ``"inf"`` strings the Rust side writes
    for infinite numeric parameters become floats again, so a loaded bank's
    ``specs`` compare equal to the dicts that built it. Only the numeric
    parameters are touched -- a feature column may be called ``"inf"``."""

    def dec(v: Any, numeric: bool) -> Any:
        if isinstance(v, dict):
            return {k: dec(x, k in _NUMERIC_KEYS) for k, x in v.items()}
        if isinstance(v, list):
            return [dec(x, numeric) for x in v]
        if numeric and v in ("inf", "-inf"):
            return math.inf if v == "inf" else -math.inf
        return v

    return dec(json.loads(text), False)


# The builders' annotations are the contract, and these two functions read
# them, so a wrong shape is reported by parameter name ("halflife must be a
# number or a list of numbers, got str '10'") before anything is serialized.
# The Rust side checks the same things, but serde cannot name the field once it
# is inside the model's tagged union, and "expected f64" with no name is not
# much of a message.


def _matches(v: Any, hint: Any) -> bool:
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        return any(_matches(v, a) for a in typing.get_args(hint))
    if hint is type(None):
        return v is None
    if origin is list:
        (inner,) = typing.get_args(hint)
        return isinstance(v, (list, tuple)) and all(_matches(x, inner) for x in v)
    if origin is dict:
        key, val = typing.get_args(hint)
        return isinstance(v, dict) and all(
            _matches(k, key) and _matches(x, val) for k, x in v.items()
        )
    if hint is bool:
        return isinstance(v, bool)
    if hint is float:
        return isinstance(v, numbers.Real) and not isinstance(v, bool)
    if hint is int:
        return isinstance(v, numbers.Integral) and not isinstance(v, bool)
    if hint is str:
        return isinstance(v, str)
    return True  # an annotation this does not read; the Rust side still checks


def _describe(hint: Any, plural: bool = False) -> str:
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        parts = [_describe(a, plural) for a in typing.get_args(hint) if a is not type(None)]
        return " or ".join(parts)
    if origin is list:
        (inner,) = typing.get_args(hint)
        return ("lists of " if plural else "a list of ") + _describe(inner, plural=True)
    if origin is dict:
        key, val = typing.get_args(hint)
        return f"a dict of {_describe(key)} -> {_describe(val)}"
    nouns = {float: ("a number", "numbers"), int: ("an int", "ints")}
    nouns |= {bool: ("a bool", "bools"), str: ("a str", "strs")}
    one, many = nouns.get(hint, (str(hint), str(hint)))
    return many if plural else one


def _got(v: Any) -> str:
    r = repr(v)
    return f"{type(v).__name__} {r if len(r) <= 60 else r[:57] + '...'}"


# The parameters whose Rust type admits ``inf`` (``Num`` / ``FloatOrList``
# rather than ``f64``): no decay, no ceiling, a pinned coefficient, no clip.
# Everywhere else an infinity is refused here by name, because once it is
# inside the model's tagged union serde can only say `expected f64`. Keyed by
# builder because ``ridge`` is a grid for ``ewridge`` and a plain float for
# ``huber``; ``"*"`` is the shared parameters. tests/test_error_messages.py
# checks this table against the Rust side.
_INF_OK: dict[str, frozenset[str]] = {
    "*": frozenset({"halflife", "min_periods", "max_dclock", "session_gap"}),
    "ewridge": frozenset({"ridge"}),
    "kalman": frozenset({"coef_halflife", "q", "revert_halflife"}),
    "sgd": frozenset({"clip_gradient", "coef_min", "coef_max"}),
    "pa": frozenset({"coef_min", "coef_max"}),
    "holt": frozenset({"trend_halflife"}),
}


def _finite(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_finite(x) for x in value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return not math.isinf(value)
    return True


def _checked[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Check each keyword against ``fn``'s annotations (and ``_common``'s for
    the shared parameters) so a wrong shape names the parameter."""
    skip = {"return", "common"}
    own = {k: v for k, v in typing.get_type_hints(fn).items() if k not in skip}
    shared = typing.get_type_hints(_common)
    shared = {k: v for k, v in shared.items() if k not in {"return", "name", "model"}}
    inf_ok = _INF_OK["*"] | _INF_OK.get(fn.__name__, frozenset())

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        name = args[0] if args else kwargs.get("name")
        if not isinstance(name, str):
            raise TypeError(f"spec name must be a str, got {_got(name)}")
        who = f"spec {json.dumps(name)}"
        for key, value in kwargs.items():
            if key == "name":
                continue
            hint = own.get(key, shared.get(key))
            if hint is None:
                raise TypeError(
                    f"{who}: {fn.__name__}() got an unexpected keyword argument {key!r}"
                )
            if not _matches(value, hint):
                raise TypeError(f"{who}: {key} must be {_describe(hint)}, got {_got(value)}")
            # Every int parameter is a count (u32 on the Rust side).
            if (
                isinstance(value, numbers.Integral)
                and not isinstance(value, bool)
                and int in (hint, *typing.get_args(hint))
                and value < 0
            ):
                raise ValueError(f"{who}: {key} must be >= 0, got {value}")
            if key not in inf_ok and not _finite(value):
                raise ValueError(f"{who}: {key} must be finite, got {_got(value)}")
        return fn(*args, **kwargs)

    return wrapper


__all__ = [
    "bocpd",
    "corrchange",
    "deco",
    "hmm",
    "rcov",
    "ew_class",
    "ew_cov",
    "ewridge",
    "ftrl",
    "holt",
    "huber",
    "kalman",
    "kmeans",
    "lasso",
    "marginal",
    "micro",
    "output_fields",
    "pa",
    "quantile",
    "rls",
    "seqtest",
    "sgd",
]


def _common(
    name: str,
    model: dict[str, Any],
    *,
    targets: list[str],
    features: list[str],
    add_intercept: bool = True,
    clock: str | None = None,
    halflife: float | list[float] | None = None,
    lam: float | None = None,
    max_dclock: float | None = None,
    on_clock_reset: str = "max",
    session: str | None = None,
    session_gap: float | str | None = None,
    weight: str | None = None,
    min_periods: float | list[float] | None = None,
    coef_every: int = 0,
    emit_sigma: bool = False,
    emit_resid_z: bool = False,
    emit_selected: bool = False,
    emit_averaged: bool = False,
    average_eta: float | None = None,
    emit_metrics: bool = False,
    conformal: float | None = None,
    conformal_rate: float | None = None,
    resid_quantiles: list[float] | None = None,
    emit_autocorr: bool = False,
    resid_autocorr_lag: int | None = None,
    emit_drift: bool = False,
    drift_delta: float | None = None,
    drift_threshold: float | None = None,
    drift_action: str = "flag",
    label_delay: float | None = None,
    group: str | None = None,
    group_close: str | None = None,
) -> dict[str, Any]:
    spec = {
        "name": name,
        "model": model,
        "targets": targets,
        "features": features,
        "add_intercept": add_intercept,
        "clock": clock,
        "halflife": halflife,
        "lam": lam,
        "max_dclock": max_dclock,
        "on_clock_reset": on_clock_reset,
        "session": session,
        "session_gap": session_gap,
        "weight": weight,
        "min_periods": min_periods,
        "coef_every": coef_every,
        "emit_sigma": emit_sigma,
        "emit_resid_z": emit_resid_z,
        "emit_selected": emit_selected,
        "emit_averaged": emit_averaged,
        "average_eta": average_eta,
        "emit_metrics": emit_metrics,
        "conformal": conformal,
        "conformal_rate": conformal_rate,
        "resid_quantiles": resid_quantiles,
        "emit_autocorr": emit_autocorr,
        "resid_autocorr_lag": resid_autocorr_lag,
        "emit_drift": emit_drift,
        "drift_delta": drift_delta,
        "drift_threshold": drift_threshold,
        "drift_action": drift_action,
        "label_delay": label_delay,
        "group": group,
        "group_close": group_close,
    }
    validate_spec(_json(spec))
    return spec


@_checked
def ewridge(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    ridge: float | list[float] | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    standardize: bool = False,
    ridge_decay: bool = False,
    coef_prior: list[list[float]] | None = None,
    session_shrink: float | None = None,
    long_halflife: float | None = None,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    window: float | None = None,
    window_every: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """EW-ridge spec (docs/PLAN.md §4.1).

    Math: EW means ``S = EW[x x^T]``, ``r_j = EW[x y_j]`` with per-row decay
    ``0.5 ** (d_clock / halflife)``; coefficients solve
    ``(S + ridge * D) beta_j = r_j`` (D = identity minus the intercept slot) on a
    schedule (``solve_every`` clock units, default halflife/50 -- every row for
    ``halflife=inf`` and for ``lam``, so set it with a large finite halflife
    or the default never comes due; ``max_rows_between_solves`` caps the
    schedule in rows, and is off by default). Predictions use the last solved
    coefficients and the state *before* the row's update.

    ``window`` puts a **hard cutoff** on the history the fit is solved from,
    in clock units: a row older than it is not in the Gram at all, where the
    exponential weight alone would leave ``0.5 ** (age / halflife)`` of it.
    Inside the window the weights are still exponential, so this is a
    *windowed exponentially weighted* regression, not a rolling OLS. It is
    exact rather than approximate: the Gram and the cross-moments are sums of
    per-row contributions, so everything at or before a time ``u`` is
    ``lam ** (t - u)`` times the accumulators as they stood then, and
    subtracting that leaves the window. The model keeps a ring of snapshots to
    do it -- the one place here where memory grows with a window rather than
    with the state -- and a halflife grid is one instance per entry, each with
    its own ring. ``window_every`` snapshots every ``n`` rows instead, which
    divides the memory and can only *shorten* the effective window.

    ``n_eff``, ``sigma`` and ``resid_z`` come from the window too, so the
    reported spread describes the rows the fit describes. Mind the solve
    schedule: the coefficients are the window's *as of the last solve*, so a
    coarse ``solve_every`` reports a window that has since moved on.
    ``window`` is refused with ``ridge_decay`` (the decaying prior's scale is
    the product of every decay factor the stream has applied, which a window
    truncates the data of but not the prior) and with ``session_shrink`` (the
    slow twin is a second accumulator under a longer halflife).

    ``ridge`` defaults to ``1e-6``. With ``standardize`` (default ``False``)
        the solve is done in correlation form and unscaled afterwards, and a
        feature whose variance is zero is dropped from the solve rather than
        blowing it up.
        ``ridge`` may be a list (one fit per value, reported side by side) and
        ``feature_sets`` names subsets of ``features``, each a fit of its own
        reported as ``pred_<t>__<set>`` -- the full set is fitted only when it
        is one of them; ``emit_selected`` then reports the fit doing best.

        ``coef_prior`` shrinks toward a stated belief instead of toward zero, one vector
        per target in the features' original units. **Whether the prior fades
        depends on ``ridge_decay``**: ``S`` is a weighted *mean*, so a plain
        ``ridge`` is a fixed per-observation penalty whose pull is permanent
        ("always stay near this belief"); with ``ridge_decay`` the prior sits on the
        decaying sum scale and fades as data arrives (the usual warm start, "begin
        at yesterday's fit and let evidence take over").

        ``session_shrink`` is a middle option between ``session_gap`` and a full
        reset (PLAN section 12 open question 1). A second accumulator tracks the
        long-run relationship at ``long_halflife``, and on a session boundary the
        two are mixed weight-respectingly::

            W'  = (1-f) * W_fast + f * W_slow
            S'  = ((1-f) * W_fast * S_fast + f * W_slow * S_slow) / W'

        so ``0`` keeps today's fit, ``1`` reverts fully to the long run, and
        anything between says "overnight, drift partway back". Unlike
        ``session_gap`` this changes what the model *believes*, not just how
        confident it is.

        Raises as every builder does (:mod:`polars_online.spec`); its own rules:
        a ``feature_sets`` entry naming a column not in ``features``, a ``coef_prior``
        vector of the wrong length, and ``session_shrink`` without
        ``long_halflife`` are ``ValueError`` naming the problem.
    """
    model: dict[str, Any] = {
        "type": "ew_ridge",
        "ridge": ridge,
        "feature_sets": [[k, list(v)] for k, v in feature_sets.items()] if feature_sets else None,
        "standardize": standardize,
        "ridge_decay": ridge_decay,
        "coef_prior": coef_prior,
        "session_shrink": session_shrink,
        "long_halflife": long_halflife,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        "window": window,
        "window_every": window_every,
    }
    return _common(name, model, targets=targets, features=features, **common)


def _numeric_keys() -> frozenset[str]:
    """Every parameter, across the builders, whose annotation admits a float."""
    keys = set()
    builders = (
        ewridge,
        rls,
        lasso,
        kalman,
        huber,
        quantile,
        ftrl,
        ew_cov,
        sgd,
        pa,
        holt,
        kmeans,
        micro,
        ew_class,
        seqtest,
        marginal,
    )
    for fn in (_common, *builders):
        for key, hint in typing.get_type_hints(getattr(fn, "__wrapped__", fn)).items():
            leaves = {hint, *typing.get_args(hint)}
            leaves |= {a for h in list(leaves) for a in typing.get_args(h)}
            leaves |= {a for h in list(leaves) for a in typing.get_args(h)}
            if float in leaves:
                keys.add(key)
    return frozenset(keys)


def output_fields(spec: dict[str, Any]) -> list[str]:
    """Struct field names this spec will produce, in order.

    ``ValueError`` for a spec that is not valid, with the builders' message
    (:mod:`polars_online.spec`); a dict that is not a spec at all is told
    what it lacks (``invalid spec: missing field `name```)."""
    return spec_output_fields(_json(spec))


def output_index(spec: dict[str, Any]) -> pl.DataFrame:
    """Every output field with the machine values its name encodes.

    One row per struct field, in order: ``field``, ``kind`` (``pred``,
    ``resid``, ``sigma``, ``n_eff``, ``coef``, ...), ``target``, ``halflife``
    (or ``lam``), ``ridge``, ``feature_set``, ``lambda`` (lasso path point),
    ``quantile``, ``columns`` (the pair an ``ew_cov`` statistic is over), and
    ``dtype`` (``f64``, ``bool``, ``str`` or ``list[f64]``).

    This is how to reach a field **without constructing its name** -- the
    string grammar (``pred_y__r0.5@h500``) stays an implementation detail::

        idx = po.spec.output_index(spec)
        name = idx.filter(
            (pl.col("kind") == "pred")
            & (pl.col("target") == "y")
            & (pl.col("ridge") == 0.5)
            & (pl.col("halflife") == 500.0)
        )["field"].item()
        out["m"].struct.field(name)

    Produced by the same Rust code that renders the names, so the metadata can
    never drift from the strings. ``ValueError`` for a spec that is not
    valid, as for :func:`output_fields`.
    """
    rows = json.loads(spec_output_index(_json(spec)))
    return pl.DataFrame(
        rows,
        schema={
            "field": pl.String,
            "kind": pl.String,
            "target": pl.String,
            "halflife": pl.Float64,
            "lam": pl.Float64,
            "ridge": pl.Float64,
            "feature_set": pl.String,
            "lambda": pl.Float64,
            "quantile": pl.Float64,
            "columns": pl.List(pl.String),
            "dtype": pl.String,
        },
    )


def coef_fields(spec: dict[str, Any]) -> pl.DataFrame:
    """Every coefficient the spec reports, in ``coef`` list order, with the
    column it becomes when the output is unnested.

    One row per (instance, target, combo, term): ``field`` (the ``coef``
    list it sits in -- ``coef``, or ``coef@h500`` per instance),
    ``position`` in that list, ``name`` (the column
    :meth:`~polars_online._frame.LazyFrameOnlineNamespace.unnest` gives it:
    ``coef_{target}_{term}{combo}{instance}``, so ``coef_y_x1__r0.5@h500``
    sits beside ``pred_y__r0.5@h500``), then ``target``, ``halflife`` (or
    ``lam``), ``ridge``, ``feature_set``, ``lambda`` (lasso path point) and
    ``term`` -- ``"intercept"``, a feature name, or ``"level"`` /
    ``"trend"`` for ``holt``. Empty for ``ew_cov`` and ``seqtest``, which
    have none.

    The names carry the ``coef_`` prefix and the target because a bare
    ``x1`` would collide with the feature column of that name in the same
    frame. To reach one coefficient in the nested output without writing
    either name::

        cf = po.spec.coef_fields(spec)
        row = cf.filter(
            (pl.col("target") == "y") & (pl.col("term") == "x1")
            & (pl.col("halflife") == 500.0)
        ).row(0, named=True)
        slope = out["m"].struct.field(row["field"]).list.get(row["position"])

    Rendered by the same Rust code as the field names, from the same slot
    order the models lay the list out in (``intercept`` first when the spec
    has one, then every feature -- zero for a feature outside a feature set
    -- per (target, combo) slot, slots in the order the ``pred`` fields
    declare them). ``ValueError`` for a spec that is not valid, as for
    :func:`output_fields`.
    """
    rows = json.loads(spec_coef_fields(_json(spec)))
    return pl.DataFrame(
        rows,
        schema={
            "field": pl.String,
            "position": pl.UInt32,
            "name": pl.String,
            "target": pl.String,
            "halflife": pl.Float64,
            "lam": pl.Float64,
            "ridge": pl.Float64,
            "feature_set": pl.String,
            "lambda": pl.Float64,
            "term": pl.String,
        },
    )


def coef_index(spec: dict[str, Any]) -> pl.DataFrame:
    """The layout of each ``coef`` list, one row per position.

    ``coef`` is flat: (target x combo) slots, each contributing its terms in
    order. This maps ``position`` -> (``target``, combo metadata, ``term``),
    where ``term`` is ``"intercept"``, a feature name, or -- for ``holt`` --
    ``"level"`` / ``"trend"``. For ``kmeans`` the slots are the centres, so
    ``target`` reads ``"cluster0"``, ``"cluster1"``, ... and ``term`` is the
    feature whose coordinate the position holds::

        ci = po.spec.coef_index(spec)
        pos = ci.filter(
            (pl.col("target") == "y") & (pl.col("ridge") == 0.5)
            & (pl.col("term") == "x1")
        )["position"].item()
        slope = out["m"].struct.field("coef@h100").list.get(pos)

    Every instance's list has this layout; :func:`coef_fields` is the same
    table per instance, with the field each list sits in and the column
    name each entry unnests to. ``ValueError`` for a spec that is not
    valid, as for :func:`output_fields`, for an ``ew_cov`` spec, which has
    no coefficients, for a ``seqtest`` spec, which emits evidence, and for a
    ``micro`` spec, whose ``coef`` has as many rows as there are
    established summaries and so no fixed layout.
    """
    kind = spec.get("model", {}).get("type")
    if kind == "ew_cov":
        msg = "ew_cov emits statistics, not coefficients"
        raise ValueError(msg)
    if kind == "seqtest":
        msg = "seqtest emits evidence (log e-values and counts), not coefficients"
        raise ValueError(msg)
    if kind == "micro":
        msg = (
            "micro's coef is one [id, label, n, radius, c_1, ..., c_p] row per established "
            "summary, as many as there are; it has no fixed layout to index"
        )
        raise ValueError(msg)
    cf = coef_fields(spec)
    first = cf["field"][0]
    return cf.filter(pl.col("field") == first).select(
        pl.col("position").cast(pl.Int64), "target", "ridge", "feature_set", "lambda", "term"
    )


@_checked
def rls(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    ridge: float | None = None,
    coef_prior: list[list[float]] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Recursive least squares spec (docs/PLAN.md section 4.2).

    Math: decayed ridge least squares solved exactly every row,
    ``A <- lam A + w z z'``, ``b_j <- lam b_j + w y_j z``, ``beta_j = A^-1 b_j``
    with ``A0 = ridge I`` and ``b0 = ridge coef_prior``. The state is the Cholesky
    factor ``R`` of ``A`` and ``u_j = R^-T b_j`` (square-root / QR form): a row
    is folded in by Givens rotations and ``beta`` read off by one
    back-substitution, O(k^2) per row with no solve staleness and none of the
    covariance recursion's drift (docs/IMPROVEMENTS.md C5). ``ridge`` sets
    ``A0 = ridge I`` (``P0 = I / ridge``; default 1.0) and (unlike ew_ridge)
    penalizes the intercept.

    Null policy deviation: a row with ANY null target is predict-only for all
    targets, because the factor ``R`` is shared across targets.
    """
    model: dict[str, Any] = {"type": "rls", "ridge": ridge, "coef_prior": coef_prior}
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def lasso(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    lasso_path: list[float],
    l1_ratio: float | None = None,
    select_halflife: float | None = None,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    window: float | None = None,
    window_every: int | None = None,
    max_cd_iters: int | None = None,
    cd_tol: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Lasso path with online lambda selection (docs/PLAN.md section 4.3).

    Math: coordinate descent on the standardized centered statistics held in the
    same accumulators as ew_ridge. For each penalty ``l`` in the (decreasing)
    ``window`` puts a **hard cutoff** on the history the path is fitted from,
    in clock units: a row older than it is not in the Gram (docs/PLAN.md §13).
    The selection error is truncated with it, so the ``lambda`` chosen is the
    one that fits the window rather than one chosen on rows the fit has
    dropped -- which means a window can change the *support*, not just the
    coefficients: a feature with no evidence inside it goes to exactly zero.
    ``window_every`` trades boundary tightness for memory, and can only
    shorten the effective window.

    ``lasso_path``, with ``C`` the feature correlation matrix and ``c`` the
    feature-target correlations::

        rho_i = c_i - sum_{j != i} C_ij b_j
        b_i   = soft(rho_i, l * l1_ratio) / (C_ii + l * (1 - l1_ratio))

    warm-started along the path and across solves, then unscaled with the
    intercept recovered as ``ybar - m . beta``. ``l1_ratio < 1`` is elastic net.

    Selection is free: predictions for every path point are computed anyway, so
    ``lam_selected_<target>`` is the argmin over the path of an EW mean squared
    out-of-sample error with halflife ``select_halflife`` (default: the model
    halflife), reported as it stood before the row -- the lambda this row was
    scored with, not the one its own error then elected. Outputs carry one
    pred/resid pair per path point.

    ``l1_ratio`` defaults to 1 (the lasso). ``solve_every`` and
    ``max_rows_between_solves`` schedule the solves as for :func:`ewridge`;
    within a solve, coordinate descent stops after ``max_cd_iters`` sweeps
    (default 100) or when no coefficient moves by more than ``cd_tol``
    (default ``1e-10``).
    """
    model: dict[str, Any] = {
        "type": "lasso",
        "lasso_path": lasso_path,
        "l1_ratio": l1_ratio,
        "select_halflife": select_halflife,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        "max_cd_iters": max_cd_iters,
        "cd_tol": cd_tol,
        "window": window,
        "window_every": window_every,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def kalman(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    coef_halflife: float | list[float],
    q: list[float] | None = None,
    obs_var: float | None = None,
    p0: float | None = None,
    share_p: bool = False,
    revert_halflife: float | list[float] | None = None,
    standardize: bool = True,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Kalman / random-walk-beta dynamic linear model (docs/PLAN.md section 4.4).

    State per target: coefficient mean ``b_j`` and covariance ``P_j``. Per row
    (clock delta ``d``, row weight ``w``)::

        b_j <- Phi b_j                 Phi = diag(2^(-d / r_i))
        P_j <- Phi P_j Phi + Q * d
        s    = z' P_j z + R_j / w
        k    = P_j z / s
        b_j <- b_j + k (y_j - z' b_j)
        P_j <- P_j - k z' P_j

    Process noise is derived from a per-factor coefficient halflife on
    standardized features: ``q_i = sigma^2 * (ln2 / h_i)^2`` (steady-state gain
    matching with EW-RLS). ``coef_halflife`` is a scalar or one value per slot
    (intercept first); ``inf`` pins that coefficient. An explicit ``q``
    overrides the derivation. Observation noise is the EW residual variance
    unless ``obs_var`` is given.

    ``p0`` is the initial coefficient covariance, ``P_0 = p0 I`` (default
    1.0). With ``standardize`` (default ``True``) the filter runs on
    standardized features, so ``coef_halflife`` and ``p0`` mean the same
    thing whatever the columns' scale; the reported coefficients are in the
    original units either way.

    ``revert_halflife`` gives each slot a reversion halflife ``r_i``: between
    observations the coefficient shrinks toward zero by ``2^(-d / r_i)``, so a
    coefficient no row has supported for a while is forgotten rather than
    carried. ``inf`` (the default) is the random walk and costs nothing. A
    scalar applies to every slot, the intercept included; a list is one value
    per slot, intercept first, and ``[inf, r, r]`` leaves the intercept a
    random walk. The pull is toward zero in the standardized coordinates when
    ``standardize`` is on -- a slope toward "no effect", the intercept toward
    "the target averages zero". With ``Q`` from ``coef_halflife`` a reverting
    slot settles at prior variance ``q_i d / (1 - phi_i^2)``, a stationary
    AR(1) instead of an unbounded walk. Predictions propagate the state by the
    same ``Phi`` for the row's clock gap.

    ``P`` is per target because the Riccati recursion depends on ``sigma^2_j``;
    ``share_p=True`` keeps one ``P`` driven by the mean ``sigma^2`` across
    targets (docs/PLAN.md marks this [validate]).

    Note ``coef_halflife`` (how fast coefficients drift) is distinct from the
    spec-level ``halflife``, which drives the standardization and residual
    variance statistics.
    """
    model: dict[str, Any] = {
        "type": "kalman",
        "coef_halflife": coef_halflife,
        "q": q,
        "obs_var": obs_var,
        "p0": p0,
        "share_p": share_p,
        "revert_halflife": revert_halflife,
        "standardize": standardize,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def huber(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    huber_delta: float | None = None,
    ridge: float | None = None,
    standardize: bool = False,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Huber regression (docs/PLAN.md section 4.5).

    IRLS reweighting on the ew_ridge update: each row's weight is scaled by the
    robust weight of its *prior* residual, so it stays out-of-sample. With
    ``d = huber_delta`` in units of the EW residual std ``s``::

        w_robust = 1            if |r| <= d * s
                 = d * s / |r|  otherwise

    The weights are per target, so ``S`` is per target here (one accumulator
    each), unlike ew_ridge which shares one. Default ``huber_delta`` is 1.5.
    ``ridge`` (default ``1e-6``), ``standardize``, ``solve_every`` and
    ``max_rows_between_solves`` mean what they mean for :func:`ewridge`.
    """
    model: dict[str, Any] = {
        "type": "huber",
        "huber_delta": huber_delta,
        "ridge": ridge,
        "standardize": standardize,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def quantile(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    quantile: float,
    ridge: float | None = None,
    standardize: bool = False,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    quantile_eps: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Quantile regression at level ``quantile`` (docs/PLAN.md section 4.5).

    The IRLS weights of the check loss, applied to the prior residual::

        w_robust = 2 * tau       * s / max(|r|, eps * s)   if r > 0
                 = 2 * (1 - tau) * s / max(|r|, eps * s)   otherwise

    ``quantile_eps`` (default ``1e-3``) floors ``|r|`` (in units of the EW
    residual std) so a near-zero residual cannot produce an unbounded weight.
    ``ridge`` (default ``1e-6``), ``standardize``, ``solve_every`` and
    ``max_rows_between_solves`` mean what they mean for :func:`ewridge`.
    """
    model: dict[str, Any] = {
        "type": "quantile",
        "quantile": quantile,
        "ridge": ridge,
        "standardize": standardize,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        "quantile_eps": quantile_eps,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def ftrl(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    alpha: float | None = None,
    beta: float | None = None,
    l1: float | None = None,
    l2: float | None = None,
    strict_binary: bool = False,
    loss: str = "logistic",
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Online regression via FTRL-proximal (docs/PLAN.md section 4.6).

    ``loss="logistic"`` (default) for binary (0/1) targets, where ``pred`` is a
    probability; ``loss="squared"`` for continuous targets, where ``pred`` is
    the linear prediction. The two differ only in the link -- the gradient is
    ``(p - y) * z`` either way -- so the squared loss gives sparse linear
    regression with no solves, and L1 support that ``ew_ridge`` does not have.

    Per-coordinate adaptive learning rates following McMahan et al. (2013),
    with the accumulators decayed on the same clock as every other model here,
    so it forgets on the same schedule::

        n_i   <- lam * n_i ;  zz_i <- lam * zz_i
        b_i   = 0 if |zz_i| <= l1 else
                -(zz_i - sign(zz_i) l1) / ((beta + sqrt(n_i)) / alpha + l2)
        p     = sigmoid(z . b)
        g_i   = (p - y) * z_i * w
        zz_i += g_i - ((sqrt(n_i + g_i^2) - sqrt(n_i)) / alpha) * b_i
        n_i  += g_i^2

    (``z`` is the row's feature vector, intercept included; ``zz`` and ``n``
    are the two per-coordinate accumulators.)
    ``pred`` is the probability from the state *before* the update, so it is
    out-of-sample like every other model, and ``resid = y - p``. Defaults
    ``alpha=0.1, beta=1.0, l1=0.0, l2=1.0``. Non-0/1 targets are clamped into
    [0, 1] unless ``strict_binary``, which skips them instead.
    """
    model: dict[str, Any] = {
        "type": "ftrl",
        "alpha": alpha,
        "beta": beta,
        "l1": l1,
        "l2": l2,
        "strict_binary": strict_binary,
        "loss": loss,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def ew_cov(
    name: str,
    *,
    features: list[str],
    stats: list[str] | None = None,
    precision_prior: float | None = None,
    mahal_quantiles: list[float] | None = None,
    pca: int | None = None,
    pca_every: int | None = None,
    lags: list[int] | None = None,
    window: float | None = None,
    window_every: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted moments of the feature columns (docs/PLAN.md 4.7).

    Not a regression: there are no targets and no coefficients, just running
    statistics of the columns you name, decayed on the same clock as every
    model here. The means and the **centred** co-moments ``C`` are kept as
    weighted means, updated in Welford's form::

        W'     = lam * W + w
        a      = lam * W / W'        b = w / W'        (a + b = 1)
        delta  = x - m
        m'     = m + b * delta
        C'_ij  = a * C_ij + a * b * delta_i * delta_j

    so ``var_i = C_ii``, ``cov_ij = C_ij`` and
    ``corr_ij = C_ij / sqrt(C_ii C_jj)`` are read off directly, and a
    variance stays accurate when the columns sit on a large offset (the raw
    ``E[x^2] - m^2`` form loses it). One O(k^2) update per row, which
    replaces the O(k^2) *passes* a pure-Polars pairwise EW correlation needs.

    ``stats`` selects which to emit, from ``mean``, ``var``, ``std``, ``cov``,
    ``corr``, ``partial_corr``, ``mahal`` and ``lagcorr`` (default: mean, std,
    corr).
    ``window`` puts a **hard cutoff** on the history, in clock units: a row
    older than ``window`` contributes nothing at all, where the exponential
    weight alone would still leave ``0.5 ** (age / halflife)`` of it -- 12.5%
    at three halflives. Inside the window the weights are *still
    exponential*, so this is not a rolling flat mean: the newest row
    dominates exactly as it does without a window.

    It is exact rather than approximate, because an exponentially weighted
    sum contains its own past: everything at or before a time ``u`` is
    ``lam ** (t - u)`` times the accumulator as it stood at ``u``, so
    subtracting that leaves precisely the rest. What the model keeps is a
    ring of snapshots, one per learned row -- the one place in this library
    where memory grows with a *window* rather than with the state, at
    ``k**2 + k + 2`` doubles each, so a 1,000-row window over 20 columns is
    about 3 MB per group. ``window_every`` snapshots every ``n`` rows instead
    and divides that by ``n``.

    Four things to know before reading the numbers:

    - **The guarantee is one-sided.** The boundary is the oldest snapshot
      still inside the window, so what is dropped is always a superset of
      what the window excludes. With ``window_every`` above 1 the effective
      window is *shorter* than asked, by at most one snapshot's spacing --
      never longer.
    - **The clock is the decayed one.** ``window`` is measured on the clock
      the decay uses, after ``max_dclock`` caps a gap and after a
      ``session_gap`` is applied, not on the raw column.
    - **The edge is a discontinuity.** A row ageing out drops its whole
      weight at once, so a windowed series has small steps an EWMA does not.
    - **It is a subtraction**, so precision falls with the fraction
      discarded: negligible at ``window = 3 * halflife`` (an eighth of the
      mass), worse as the window shortens toward the halflife.

    ``n_eff`` becomes the weight *inside* the window, which stops growing
    once the window fills, so ``min_periods`` gates on a quantity with a
    ceiling. A clock gap longer than ``window`` empties it and the row
    reports nulls rather than stale numbers. ``window`` does not combine with
    ``lags`` or ``mahal_quantiles``, which accumulate over a history it does
    not truncate; both are refused by name.

    For moments on data that fits in memory, polars already does this --
    ``df.rolling(clock, period=...).agg(...)`` with an exponential weight
    agrees to 1e-14. The reason to reach for the spec is a stream, a saved
    state, or the cost: polars recomputes each window at ``O(n*W)`` where
    this is ``O(n)``.

    ``stats=[]`` emits nothing but ``n_eff`` and accumulates all the same
    (docs/ENHANCEMENTS.md E43): the spec's value is then its state -- the
    Gram read back with :meth:`ModelBank.gram`, the moments with
    :meth:`ModelBank.describe` -- which is the form for a wide set of columns,
    where emitting even the means is k values per row nobody reads.
    ``partial_corr`` is the correlation between two columns *controlling for
    every other column*, read off the precision matrix as
    ``-P_ij / sqrt(P_ii P_jj)``. It needs ``precision_prior``: the precision
    matrix is ``(C + s * prior * I)^-1``, solved from the co-moments on each
    row it is read (O(k^3), only when asked for); like RLS's ``P0`` the prior
    fades as data accumulates.

    Pairwise statistics are emitted for each unordered pair ``i < j``, named
    after the columns (``corr_x0_x1``, ``pcorr_x0_x1``).

    ``mahal`` (docs/ENHANCEMENTS.md E37) is the row's Mahalanobis distance
    from the decayed history, one field over all the columns::

        mahal = sqrt((x - m)^T (C + s * prior * I)^-1 (x - m))

    in units of standard deviations, like ``resid_z`` for a regression: with
    ``k`` Gaussian columns ``mahal^2`` is chi-squared with ``k`` degrees of
    freedom, so ``mahal^2 > chi2.ppf(0.99, k)`` flags a 1% outlier. One
    Cholesky solve per row (O(k^3)), only when asked for; it needs
    ``precision_prior``. ``mahal_quantiles`` adds a P-squared running
    quantile of the past scores per level (``mahal_q0.99``), a threshold
    from the stream's own history instead of a table; the row's own score
    joins after it is read.

    ``pca=r`` (docs/ENHANCEMENTS.md E38) tracks the top ``r`` principal
    components of the covariance: per component ``j`` the fields
    ``pc<j>_var`` (its eigenvalue), ``pc<j>_share`` (of the total variance),
    ``pc<j>_<feature>`` (its unit loading on each column, largest entry
    positive) and ``pc<j>_score`` (the row's coordinate ``v_j . (x - m)``).
    The eigendecomposition is refreshed every ``pca_every`` learned rows
    (default 1, O(k^3) each) after the row is folded in; between refreshes
    the loadings are frozen, so a row's scores never depend on chunking.

    ``lags`` accumulates **lagged** cross-moments beside the contemporaneous
    ones (ENHANCEMENTS E56): with ``W`` and ``m`` the weight and mean before
    the row, and both deviations taken against that mean,

    .. code-block:: text

        C_l' = a * C_l + a * b * (x_t - m) (x_{t-l} - m)'

    which is the same ``a`` and ``b`` the co-moments use -- so lag 0 would be
    ``comoments`` exactly. Lags are counted in **learned rows within the
    group**, not clock units, and must be strictly increasing and ``>= 1``;
    the list order is the output order. The ring of past rows is emptied on a
    session change and on a clock gap beyond ``max_dclock``, the two events
    after which "the row `l` back" no longer means a row `l` ago; a
    zero-weight row ages the matrices and does not enter the ring.

    Read them from :meth:`ModelBank.gram` as ``lags`` and ``lag_comoments``
    (an ``(L, k, k)`` array), or add ``"lagcorr"`` to ``stats`` to emit
    ``lagcorr_<a>_<b>_l<l>`` = ``C_l[a,b] / sqrt(C_0[a,a] * C_0[b,b])`` per
    lag and **ordered** pair, the auto terms included: ``k**2`` slots a lag,
    because the lagged matrix is not symmetric.

    Values are read from the state *before* each row, so an ``ew_cov`` output
    can be used as a feature for that same row without leaking it.
    """
    model: dict[str, Any] = {
        "type": "ew_cov",
        "stats": stats,
        "precision_prior": precision_prior,
        "mahal_quantiles": mahal_quantiles,
        "pca": pca,
        "pca_every": pca_every,
        "lags": lags,
        "window": window,
        "window_every": window_every,
    }
    if "targets" in common:
        msg = (
            f"spec {json.dumps(name)}: ew_cov() takes no targets; its statistics are over "
            "the features"
        )
        raise TypeError(msg)
    # `targets` is required by the common-parameter schema but unused here.
    return _common(name, model, targets=[features[0]], features=features, **common)


@_checked
def sgd(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    loss: str = "squared",
    huber_delta: float | None = None,
    quantile: float | None = None,
    eps: float | None = None,
    learning_rate: float | None = None,
    schedule: str = "constant",
    power: float | None = None,
    l2: float | None = None,
    clip_gradient: float | None = None,
    scale_features: bool = False,
    coef_min: float | list[float] | None = None,
    coef_max: float | list[float] | None = None,
    coef_sum: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Stochastic gradient descent with pluggable losses (ENHANCEMENTS E16).

    The cheap baseline: one gradient step per row, no solves, O(k) rather than
    O(k^2). Also the only model here that takes **count targets**, via
    ``loss="poisson"`` with a log link -- none of the exact solvers cover those.

    With ``eta = z . b``, ``p = link(eta)`` and ``d = dL/d(eta)``:

    ==================== ========= ============ ==============================
    loss                 link      ``p``        ``d``
    ==================== ========= ============ ==============================
    squared              identity  ``eta``      ``p - y``
    huber                identity  ``eta``      ``clamp(p - y, +/-delta)``
    quantile             identity  ``eta``      ``1{y < p} - tau``
    epsilon_insensitive  identity  ``eta``      0 inside the tube, else sign
    poisson              log       ``exp(eta)`` ``p - y``
    logistic             sigmoid   sigmoid      ``p - y``
    ==================== ========= ============ ==============================

    then ``g_i = d * z_i * w + l2 * b_i`` and ``b_i -= lr_i * g_i``.

    ``schedule`` is ``constant``, ``inv_scaling`` (``lr / (1 + n_eff)**power``)
    or ``adagrad`` (``lr / (sqrt(G_i) + 1e-8)``). AdaGrad's accumulator and
    ``n_eff`` both decay on the model's clock, so an annealed or adapted rate
    re-opens after a long gap instead of staying frozen.

    ``clip_gradient`` defaults to ``1e3`` rather than being off. Identity-link
    losses do not need it, but ``poisson`` does: ``p = exp(eta)``, so a row that
    pushes ``eta`` up makes the next gradient exponentially larger and a
    constant rate diverges within a few thousand rows (measured: the intercept
    ran to -4e10 unclipped, and recovered the true value clipped). The cap does
    not bind for ordinary squared-loss fits.

    Note ``epsilon_insensitive`` has a sign-valued subgradient, so a constant
    rate oscillates in a band around the optimum; use ``inv_scaling`` with it.
    ``huber_delta`` here is in **target units**, unlike the ``huber`` model
    where it is in units of the residual std.

    Defaults: ``learning_rate=0.01``, ``l2=0.0``, ``huber_delta=1.0``,
    ``quantile`` must be given for that loss, ``eps=0.1`` (the half-width of
    the insensitive tube, in target units), ``power=0.5`` for
    ``inv_scaling``. ``scale_features`` (default ``False``) takes the step in
    standardized coordinates, which is the difference between one learning
    rate for every column and one per scale.

    **Constrained coefficients** (ENHANCEMENTS E40). ``coef_min`` and
    ``coef_max`` bound each slope (one number for every feature, or a list
    with one entry per feature; ``-inf`` / ``inf`` for no bound) and
    ``coef_sum`` fixes the slopes' total; the intercept is always free. After
    each update the slopes are replaced by the nearest point of the feasible
    set (the Euclidean projection): ``b_i = clamp(b_i - mu, lo_i, hi_i)``
    with ``mu = 0`` for a box alone and, with a sum, the one ``mu`` at which
    the sum holds -- found exactly by sorting the ``2k`` breakpoints where a
    coordinate meets a bound. ``coef_min=0, coef_sum=1`` puts the slopes on
    the simplex: portfolio weights, mixing weights, an ensemble over
    forecasts. The starting point (all zero) is projected too, so a simplex
    starts uniform. With ``scale_features`` the step is taken in standardized
    coordinates and so is the projection, with the bounds and the sum
    carried over exactly (``b_i = c_i * scale_i``); the reported coefficients
    satisfy the constraint in the caller's units after every learned row. A
    sum the bounds cannot reach, a floor above a cap, or an infinite bound on
    the wrong side (``inf`` as a floor) is refused by name; floors of
    ``[0.1, 0.2, 0.3]`` accept a sum of ``0.6`` although they add up to
    ``0.6000000000000001``.
    """
    model: dict[str, Any] = {
        "type": "sgd",
        "loss": loss,
        "huber_delta": huber_delta,
        "quantile": quantile,
        "eps": eps,
        "learning_rate": learning_rate,
        "schedule": schedule,
        "power": power,
        "l2": l2,
        "clip_gradient": clip_gradient,
        "scale_features": scale_features,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "coef_sum": coef_sum,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def pa(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    mode: str = "pa1",
    c: float | None = None,
    eps: float | None = None,
    coef_min: float | list[float] | None = None,
    coef_max: float | list[float] | None = None,
    coef_sum: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Passive-aggressive regression (ENHANCEMENTS E17; Crammer et al. 2006).

    Each row poses a constraint -- "get within ``eps`` of this target" -- and the
    update is the *smallest* change to the coefficients that satisfies it.
    Passive when the constraint already holds, aggressive when it does not, and
    there is no learning rate to tune. With ``p = z . b``,
    ``loss = max(0, |y - p| - eps)`` and ``s = ||z||^2``::

        pa    tau = loss / s                 (unbounded)
        pa1   tau = min(c, loss / s)         (capped at c)
        pa2   tau = loss / (s + 1 / (2c))    (damped by c)
        b    += tau * sign(y - p) * z

    ``mode`` is ``"pa"``, ``"pa1"`` (the default) or ``"pa2"``; ``c``
    defaults to 1.0 and ``eps`` to 0.1, in target units.

    A row weight below 1 scales ``tau``, so a half-weight row moves the fit
    half as far; a weight above 1 counts as 1. The update is a projection onto
    the row's constraint, and repeating a projection changes nothing, so there
    is no "two observations" to emulate -- scaling past the projection would
    overshoot it.

    **Decay note.** Unlike the other models, PA keeps no accumulators, so there
    is nothing for the clock to decay: each step fully satisfies the current
    row and older rows survive only through the coefficients they left behind.
    ``n_eff`` still decays so ``min_periods`` means the same thing as
    elsewhere, but the coefficients have no half-life. Prefer ``pa1``/``pa2``
    (the default is ``pa1``) when outliers are possible: plain ``pa`` will move
    the fit as far as it takes to satisfy a single bad row.

    ``coef_min``, ``coef_max`` and ``coef_sum`` constrain the slopes exactly
    as for :func:`sgd` (ENHANCEMENTS E40): the projection follows each
    update, so the step no longer satisfies the row's margin exactly -- it is
    the closest feasible coefficient to the one that would. A truth outside
    the feasible set is never realizable, so PA keeps stepping against the
    walls; a small ``c`` keeps those steps small.
    """
    model: dict[str, Any] = {
        "type": "pa",
        "mode": mode,
        "c": c,
        "eps": eps,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "coef_sum": coef_sum,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def holt(
    name: str,
    *,
    targets: list[str],
    level_halflife: float | None = None,
    trend_halflife: float | None = None,
    features: list[str] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Holt's linear trend method (ENHANCEMENTS E25).

    Level plus slope, **no features** -- the forecasting baseline a
    feature-based model should have to beat. If a regression cannot outperform
    "the series is going up at about this rate", the features are not earning
    their place.

    Per row, with clock delta ``d`` and halflife-derived rates
    ``alpha = 1 - 0.5**(d/level_halflife)`` and likewise ``beta``::

        pred = l + b * d                      (extrapolate d clock units ahead)
        l'   = alpha * y + (1 - alpha) * pred
        b'   = beta * (l' - l) / d + (1 - beta) * b

    Deriving the rates from halflives keeps the parameter meaning identical to
    every other model here, so an irregular clock forecasts the right distance
    ahead instead of treating every row as one step. ``level_halflife``
    defaults to the spec's ``halflife`` and ``trend_halflife`` to four times
    that; ``trend_halflife=inf`` pins the trend, giving a plain EW level.

    ``coef`` is ``[level, trend]`` per target -- the whole state.
    """
    model: dict[str, Any] = {
        "type": "holt",
        "level_halflife": level_halflife,
        "trend_halflife": trend_halflife,
    }
    return _common(name, model, targets=targets, features=features or [], **common)


@_checked
def kmeans(
    name: str,
    *,
    features: list[str],
    k: int,
    warm_rows: int | None = None,
    seed_rule: str | None = None,
    seed: int | None = None,
    update_every: int | None = None,
    split_merge: float | None = None,
    split_merge_every: int | None = None,
    dead_frac: float | None = None,
    standardize: bool | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted k-means over the feature columns
    (docs/CLUSTERING.md section 6.2; docs/PLAN.md section 11a).

    Not a regression: there are no targets. Each row is assigned to the
    nearest centre *before* it is learned, so the outputs are out-of-sample
    like every prediction here. Per instance the struct holds ``cluster``
    (``i32``, the nearest centre's index), ``dist`` (the distance to it),
    ``dist2`` (the distance to the runner-up; null when ``k == 1``),
    ``n_eff`` and ``coef`` (the centres, ``k`` rows of ``len(features)``
    flattened, cluster-major -- :func:`coef_index` lays it out).

    Every centre is the decayed weighted mean of the rows assigned to it,
    the same recursion as ``ew_cov``'s mean::

        n'_j  = lam * n_j + w
        c'_j  = (lam * n_j * c_j + w * x) / n'_j       for the nearest j

    and, alongside it, the EW squared radius ``r2_j`` (the mean of
    ``|x - c_j|^2`` over the rows assigned there) that the split-merge rule
    reads. Rows are folded into per-centre batches and applied every
    ``update_every`` learned rows, so ``update_every=1`` is plain sequential
    k-means and a larger value a mini-batch one.

    ``standardize`` (default ``True``) measures distances in units of each
    feature's EW standard deviation, tracked alongside the centres; the
    coordinates themselves are never rescaled, so the centres stay in the
    features' units.

    **Seeding.** The first ``max(warm_rows, k)`` learned rows (default 500)
    are buffered, then the centres are placed by ``seed_rule``: ``"lloyd"``
    (default: k-means++ then ten weighted Lloyd iterations over the buffer),
    ``"kmeanspp"``, ``"farthest"`` (Gonzalez, from the first row) or
    ``"first"`` (the first ``k`` distinct rows). ``seed`` (default 0) drives
    the two random rules; the same seed gives the same centres. The buffer
    is replayed into the centres and freed, so the model is O(state) again
    from that row on. Outputs are null until seeding and until ``n_eff``
    reaches ``min_periods``.

    **Split-merge** (``split_merge``, default 0.5; ``0`` disables). A row
    farther from its centre than a blob of the typical radius produces (about
    four standard deviations of ``|x - c|^2`` above it) is *far*: it is
    scored, but summarised instead of learned, so it neither drags the centre
    nor widens the radius. Every ``split_merge_every`` learned rows (default 100) the
    two closest centres are compared: if their distance is under
    ``split_merge`` times the sum of their radii, and enough far rows have
    gathered somewhere (at least three, and five per cent of the window's
    weight), they are merged and the freed centre is placed at the far rows'
    mean, so a cluster that appears after seeding gets a centre without
    anyone restarting. ``dead_frac`` (default 0.05; ``0`` disables) re-places
    a centre whose weight has decayed below ``dead_frac * n_eff / k`` the
    same way, on whatever far rows there are: a centre whose cluster vanished
    is re-placed ``log2(1 / dead_frac)`` halflives later (4.3 at the default,
    2 at 0.25), and a cluster lighter than ``dead_frac / k`` of the stream
    loses its centre whenever any row is far. Far rows still count in the
    radius at each check as if they sat at the cut, so a cut the data has
    outgrown widens until the rows are learned again. Seeding leaves rows far
    from the buffer's mean out of the choice of seeds by the same rule.

    Values are read from the state *before* each row, so a ``kmeans`` output
    can be used as a feature for that same row without leaking it. Nothing
    residual-based applies (``emit_sigma``, ``emit_metrics``, drift, ...);
    each is refused by name.
    """
    model: dict[str, Any] = {
        "type": "kmeans",
        "k": k,
        "warm_rows": warm_rows,
        "seed_rule": seed_rule,
        "seed": seed,
        "update_every": update_every,
        "split_merge": split_merge,
        "split_merge_every": split_merge_every,
        "dead_frac": dead_frac,
        "standardize": standardize,
    }
    if "targets" in common:
        msg = (
            f"spec {json.dumps(name)}: kmeans() takes no targets; its clusters are over "
            "the features"
        )
        raise TypeError(msg)
    # `targets` is required by the common-parameter schema but unused here.
    return _common(name, model, targets=[features[0]], features=features, **common)


@_checked
def micro(
    name: str,
    *,
    features: list[str],
    eps: float,
    beta_mu: float | None = None,
    max_clusters: int | None = None,
    prune_every: int | None = None,
    macro_link: float | None = None,
    standardize: bool | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Density-based clustering over the feature columns: DenStream-style
    micro-clusters with a linkage macro step (docs/CLUSTERING.md section
    6.5; docs/PLAN.md section 11a).

    Not a regression: there are no targets, and unlike :func:`kmeans` there
    is no fixed number of clusters -- the model keeps a bounded set of small
    summaries (micro-clusters), each a decayed weight, centre and radius,
    and reads clusters off them as the chains of summaries that touch. That
    finds clusters of any shape (moons, rings), reports rows that belong to
    none, and lets clusters appear and vanish as the stream moves.

    Per instance the struct holds, all read *before* the row is learned:

    - ``cluster`` (``i64``): the label of the nearest established summary,
      null while there is none;
    - ``dist``: the distance to that summary's centre;
    - ``micro`` (``i64``): the id of the summary this row goes to -- the one
      it opens, when none can take it;
    - ``outlier`` (``bool``): whether no established summary takes it;
    - ``n_clusters`` and ``n_micro`` (``i32``): live clusters and live
      summaries, so churn is visible without diffing labels;
    - ``n_eff``, and ``coef``: the established summaries, one
      ``[id, label, n, radius, c_1, ..., c_p]`` row each, flattened (as many
      rows as there are, so :func:`coef_index` does not apply).

    Ids are monotone and never reused; a label is the smallest id in its
    chain, so it survives everything but the loss of that summary.

    **Math.** A summary is ``(n, c, r2)``: decayed weight, centre, and the
    EW mean squared distance of its rows from the centre (DenStream's
    radius, with the fading function being the decay). Distances are
    measured in the metric ``mw_i = 1 / var_i`` when ``standardize`` (the
    default), so ``eps`` is a bound per standardized coordinate and the
    bound in the metric is ``E = eps^2 p`` for ``p`` features. Each row::

        n_j <- lam n_j                                   every summary
        j   = the nearest potential summary, if absorbing a unit row keeps
              a r2_j + a b |x - c_j|^2 <= E,   a = n_j/(n_j+1), b = 1/(n_j+1)
              else the nearest outlier summary by the same test
              else a new summary at x with the next id
        n_j <- n_j + w,  c_j <- c_j + w/n_j (x - c_j),  r2_j <- min(., E)

    A summary is *potential* (established) once ``n >= beta_mu`` (default
    3) and *outlier* below; a new one is opened at the cap ``max_clusters``
    (default 200) by evicting the lightest outlier summary, else the
    lightest potential one. Every ``prune_every`` learned rows (default
    100) a checkpoint drops potential summaries lighter than ``beta_mu``
    and outlier summaries lighter than DenStream's ``xi(age)`` -- the
    weight a summary that had been gathering a row per clock unit since it
    opened would need to reach ``beta_mu`` within ``Tp`` more, with
    ``Tp = ceil(h log2(beta_mu / (beta_mu - 1)))`` for halflife ``h``; with
    no decay nothing is pruned, only capped -- and then links the potential
    summaries by single linkage: centres within ``L`` of each other share
    a label. ``L`` is ``macro_link`` times ``eps sqrt(p)`` when given
    (``0`` links nothing, so each summary is its own cluster; ``2`` links
    summaries that touch), and otherwise derived at each checkpoint from
    the spacing the summaries already show -- 1.5 times the 90th percentile
    of the nearest-neighbour distance, never below ``2 eps sqrt(p)`` -- so
    that a chain along a shape holds without a constant that fragments one
    shape and bridges another.

    **Choosing eps.** ``eps`` is the within-cluster spread the model should
    read as *one* cluster, per standardized coordinate: about 0.07 for
    two-dimensional shapes, 0.3 for well-separated Gaussians in twenty
    dimensions. Two failures are silent and read off the outputs: if
    nearly every row is an ``outlier`` and ``cluster`` stays null, ``eps``
    is too small (no summary reaches ``beta_mu`` before it is pruned); if
    ``n_micro`` is about the number of clusters, ``eps`` is too coarse for
    the derived link, which then bridges them -- lower ``eps``, or set
    ``macro_link`` (2 links only summaries that touch).

    A row of weight ``w`` is admitted where a unit row would be and
    absorbed with its full weight; a zero-weight row advances the clock and
    learns nothing. Values are read from the state *before* each row, so a
    ``micro`` output can be used as a feature for that same row without
    leaking it. Nothing residual-based applies (``emit_sigma``,
    ``emit_metrics``, drift, ...); each is refused by name.
    """
    model: dict[str, Any] = {
        "type": "micro",
        "eps": eps,
        "beta_mu": beta_mu,
        "max_clusters": max_clusters,
        "prune_every": prune_every,
        "macro_link": macro_link,
        "standardize": standardize,
    }
    if "targets" in common:
        msg = (
            f"spec {json.dumps(name)}: micro() takes no targets; its clusters are over the features"
        )
        raise TypeError(msg)
    return _common(name, model, targets=[features[0]], features=features, **common)


@_checked
def ew_class(
    name: str,
    *,
    features: list[str],
    label: str,
    classes: list[str],
    covariance: str | None = None,
    precision_prior: float,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Class-conditional Gaussian classifier -- quadratic discriminant
    analysis, linear discriminant analysis or Gaussian naive Bayes -- on the
    EW moments :func:`ew_cov` keeps, one set per class (docs/PLAN.md section
    11a, Task 27).

    Not a regression: ``label`` names the column that holds the class of each
    row, and ``classes`` lists every value it can hold (``targets`` is not a
    keyword). The label column is read as a key -- any dtype with a string
    form, so ``["0", "1"]`` for an integer column and ``["true", "false"]``
    for a boolean one -- and a non-null value it does not list is an error
    naming the row. A null label is a row to score but not to learn from:
    the model classifies it and ticks its clock, and no class moves.

    Per instance the struct holds, all read *before* the row is learned:

    - ``class`` (``str``): the class with the largest posterior (the first,
      on a tie), null before ``min_periods`` and while no class has been
      seen;
    - ``p_<class>`` for each class in ``classes`` order: its posterior
      probability, so the ``p_`` fields sum to 1 -- exactly ``0`` for a
      class no row has carried yet;
    - ``n_eff``, and ``coef``: the class means, one row per class in
      ``classes`` order, each one entry per feature (:func:`coef_index` lays
      the list out as ``(class, feature)``; a class not yet seen is null).

    **Math.** Each class ``c`` keeps an EW weight ``n_c``, mean ``mu_c`` and
    covariance ``C_c`` -- :func:`ew_cov`'s own weighted Welford recursion
    over the rows labelled ``c``, every other class decaying by the same
    ``lam``. Before a row ``x`` is learned it is scored against every seen
    class::

        pi_c = n_c / sum_c' n_c'
        r_c  = precision_prior * s_c         s_c: the class's prior scale
        M_c  = C_c + r_c I                                  ("full", QDA)
        M    = sum_c pi_c (C_c + r_c I)                     ("shared", LDA)
        M_c  = diag(C_c) + r_c I                        ("diagonal", naive Bayes)
        l_c  = ln pi_c - 1/2 ln det M_c - 1/2 (x - mu_c)' M_c^-1 (x - mu_c)
        p_c  = exp(l_c - max_c' l_c') / sum_c' exp(l_c' - max l)

    ``precision_prior`` is a ridge on every class covariance, in the units of
    the features, so the first rows of a class -- whose sample covariance is
    singular -- are scored with a finite, isotropic one. It is scaled by
    ``s_c``, the class's own prior scale, which starts at ``1`` and decays by
    ``lam * n_c / (lam * n_c + w)`` on every row the class learns, so the
    ridge washes out as the class fills in (exactly :func:`ew_cov`'s
    ``precision_prior``). ``covariance`` is ``"full"`` (the default), a
    covariance per class; ``"shared"``, the weight-averaged one, so the
    decision boundaries are linear; or ``"diagonal"``, the variances alone.
    Then the labelled row's class learns it::

        n_c   <- lam n_c + w
        mu_c  <- mu_c + (w / n_c) (x - mu_c)
        C_c   <- weighted Welford on (x - mu_c_old)(x - mu_c_new)'

    and ``n_eff <- lam n_eff + w`` counts every accepted row, labelled or
    not, so ``min_periods`` means the same number of rows as everywhere
    else. A row with a non-finite feature is null and learns nothing; a
    zero-weight row advances the clock. Values are read from the state
    *before* each row, so a ``p_<class>`` can be a feature for that same row
    without leaking it. Nothing residual-based applies (``emit_sigma``,
    ``emit_metrics``, ``conformal``, drift, ...); each is refused by name.
    """
    model: dict[str, Any] = {
        "type": "ew_class",
        "classes": classes,
        "covariance": covariance,
        "precision_prior": precision_prior,
    }
    if "targets" in common:
        msg = f"spec {json.dumps(name)}: ew_class() takes `label`, not targets"
        raise TypeError(msg)
    return _common(name, model, targets=[label], features=features, **common)


@_checked
def seqtest(
    name: str,
    *,
    targets: list[str],
    a: str | None = None,
    b: str | None = None,
    a_suffix: str | None = None,
    b_suffix: str | None = None,
    features: list[str] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Sequential test of a sign by betting -- an e-process, read at any row
    (ENHANCEMENTS E42, Task 30).

    Per target the row's value is reduced to its sign ``s`` in ``{-1, 0, +1}``
    and two bettors play it, one per direction. With ``n_pos`` and ``n_neg``
    the counts of positive and negative rows *before* this one and
    ``n = n_pos + n_neg``::

        lam_pos = max(0, (n_pos - n_neg) / (n + 1))    lam_neg = max(0, (n_neg - n_pos) / (n + 1))
        E_pos  *= 1 + lam_pos * s                     E_neg  *= 1 - lam_neg * s

    ``(n_pos - n_neg) / (n + 1)`` is ``2p - 1`` for the Krichevsky-Trofimov
    estimate ``p = (n_pos + 1/2) / (n + 1)`` of ``P(s = +1)``: the stake a
    gambler with a ``Beta(1/2, 1/2)`` prior puts on the next sign, clipped so
    that each side bets only on the direction it tests. Both stakes are
    computed from the rows before, so under its null -- given everything
    before it, a row is at least as likely to be negative as positive --
    ``E_pos`` is a non-negative supermartingale with ``E_pos[0] = 1``, and
    Ville's inequality gives ``P(max_t E_pos[t] >= 1/alpha) <= alpha``.
    That is the whole guarantee: ``log_e_pos >= log(1/alpha)`` on *any* row
    rejects "no more positives than negatives" at level ``alpha``, the
    stream can be read at every row and stopped the moment it crosses, and
    nothing about ``y`` but its sign is assumed -- no independence of the
    sizes, no bound, no variance. What it does not test is the size: a
    stream up by a hair 60% of the time and down by a mile the rest rejects.
    ``(E_pos + E_neg) / 2`` is the two-sided e-value.

    The struct holds, per target ``t`` and read *before* the row is learned:
    ``log_e_pos_<t>`` and ``log_e_neg_<t>`` (``log E``, so ``0`` is no
    evidence and ``log(20) = 3.0`` is level ``0.05``), ``n_pos_<t>`` and
    ``n_neg_<t>`` (``Int64`` counts), and ``n_eff``. A zero or null target
    bets nothing and counts nothing. ``weight`` is refused: every learned
    row is one trial. There is no ``halflife``/``lam``: a process that forgot its
    losses would not be an e-process; ``session`` or ``on_clock_reset =
    "reset_state"`` restarts it. ``min_periods`` defaults to 0. No
    ``features`` (the column is the test; the keyword is taken so that a
    frame namespace can pass ``[]``), no ``coef``, and nothing residual-based
    applies -- there is no prediction.

    **Comparing two specs.** Given ``a`` and ``b``, the names of two other
    specs of the same bank, each target ``t`` names a residual field both
    carry -- ``resid_<t>`` on each side, plus the side's grid suffix when it
    is a grid (``a_suffix="@h50"`` picks ``resid_<t>@h50`` of ``a``;
    ``"__r0.1"`` a ridge instance) -- and the sign tested is that of
    ``|resid_b| - |resid_a|``: positive when ``a`` was closer on the row.
    Any loss that grows with ``|resid|`` (squared, absolute, Huber) gives the
    same sign, so this is a test of "``a`` beats ``b``" under any of them,
    and the fields read ``log_e_a_<t>``, ``log_e_b_<t>``, ``wins_a_<t>``,
    ``wins_b_<t>``, ``n_eff``. The bank runs ``a`` and ``b`` first, so the
    comparison reads the same out-of-sample residuals their structs report;
    a row where either side is null (warm-up, a skipped row) is a row the
    test sits out. A spec named against itself with the same suffix (two
    instances of one grid, ``a_suffix="@h20"`` against ``b_suffix="@h400"``,
    is a comparison like any other), a side that is not in the bank or is
    itself a ``seqtest``, and a target neither side has a residual for are
    refused by name. ``po.eval.seqtest`` is the same
    computation in polars expressions over a frame in memory.
    """
    model: dict[str, Any] = {
        "type": "seqtest",
        "a": a,
        "b": b,
        "a_suffix": a_suffix,
        "b_suffix": b_suffix,
    }
    return _common(name, model, targets=targets, features=features or [], **common)


@_checked
def marginal(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Every (feature, target) pair's exponentially weighted moments, kept in
    the state and read back as a frame (ENHANCEMENTS E44, Task 37).

    Not a regression and not a joint fit: each pair ``(x_j, y_t)`` is its own
    two-column ``ew_cov``, so a wide feature set against a few targets costs
    ``O(p * T)`` per row rather than the ``O((p + T)^2)`` of one ``ew_cov``
    over all the columns. Per target ``t`` on a row where ``y_t`` is present,
    with ``W_t`` the target's accumulated weight before the row::

        W'_t = lam * W_t + w        a = lam * W_t / W'_t        b = w / W'_t
        Q'_t = lam^2 * Q_t + w^2
        S'_yy      = a * S_yy      + a * b * (y_t - m_y)^2
        S'_xx[t,j] = a * S_xx[t,j] + a * b * (x_j - m_x[t,j])^2
        S'_xy[t,j] = a * S_xy[t,j] + a * b * (x_j - m_x[t,j]) * (y_t - m_y)
        m'  = m + b * (value - m)                     for m_y and each m_x[t,j]

    (deviations from the means *before* the row), the same arithmetic as
    ``ew_cov``'s, so a pair's ``corr`` equals the ``corr`` an ``ew_cov`` over
    the two columns reports, to the bit. A null target ages that target
    (``W_t *= lam``, ``Q_t *= lam^2``) and learns nothing for it; a null
    feature skips the row, as everywhere. Weights, ``halflife``/``lam``, a
    ``clock``, ``session`` and ``group`` apply as they do to every model.

    Nothing is emitted per row but ``n_eff``: the pairs live in the state and
    :meth:`ModelBank.marginal` reads them as a long frame, one row per
    (group, instance, feature, target) with ``n_eff`` (the target's ``W_t``),
    ``n_kish`` (``W_t^2 / Q_t``, the Kish effective sample size -- the row
    count that carries the same information as the weights do; ``(1 + lam) /
    (1 - lam)`` in the limit for unit weights), ``mean_x``, ``var_x``,
    ``mean_y``, ``var_y``, ``cov``, and the derived ``corr`` (``cov / sqrt(var_x
    var_y)``), ``beta`` (``cov / var_x``, the slope of ``y`` on ``x`` alone)
    and ``t`` (``corr * sqrt((n_kish - 2) / (1 - corr^2))``, the t-statistic
    of that correlation at the Kish sample size -- a descriptive scale, not a
    p-value, since the rows are neither independent nor Gaussian). The three
    derived values are null until the target's ``W_t`` reaches ``min_periods``
    (default 3: two rows give a correlation of ±1 whatever the data, three
    the first one with content), and ``var_x = 0`` or ``n_kish <= 2`` leave
    the ones they undefine null.

    ``add_intercept`` and ``coef_every`` have nothing to act on here, and
    nothing residual-based applies (there is no prediction), so the residual
    switches are refused by name. A column may not be both a target and a
    feature.
    """
    model: dict[str, Any] = {"type": "marginal"}
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def deco(
    name: str,
    *,
    features: list[str],
    dynamics: str = "ew",
    alpha: float | None = None,
    beta: float | None = None,
    blocks: dict[str, list[str]] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Dynamic equicorrelation: one number for the whole correlation matrix,
    ``O(m)`` a row (Engle & Kelly 2012; ENHANCEMENTS E55, Task 46).

    A correlation matrix of ``m`` series has ``m(m-1)/2`` free entries, and a
    stream cannot keep them all moving without ``O(m^2)`` a row. DECO
    replaces them with their average and estimates *that*. The row is
    standardised against the pre-row means and variances of an ``EwDiag``,
    ``r_i = (x_i - m_i)/sqrt(v_i)``, and with ``S1 = sum(r)`` and
    ``S2 = sum(r*r)`` over the ``n`` features the row's estimate is their
    Lemma 2.3::

        u = (S1**2 - S2) / ((n - 1) * S2)      = mean_{i!=j} r_i r_j / mean_i r_i**2

    which lies in ``(-1/(n-1), 1)``. The level then follows one of two
    dynamics, on the model's own clock with decay factor ``lam`` and row
    weight ``w``::

        "ew":      W' = lam*W + w,  a = lam*W/W',  b = w/W'
                   rho' = a*rho + b*u                       (rho = u at W = 0)
        "linear":  rho' = (1 - alpha - beta)*rho_bar' + alpha*u + beta*rho

    where ``rho_bar`` is the ``"ew"`` recursion run alongside as the target
    of the linear one. ``"linear"`` needs ``alpha`` and ``beta``, both
    ``>= 0`` with ``alpha + beta < 1``; ``"ew"`` refuses them.
    ``halflife``/``lam`` are required either way -- both the standardiser and
    the level decay on them.

    Two departures from the paper, on purpose. Its eq. 21 has a free
    intercept and it applies correlation targeting to the DCC ``Q``
    recursion, not to the linear one; writing the intercept as
    ``(1 - alpha - beta) * rho_bar`` is our reparameterisation, chosen
    because a streaming model has no sample on which to fit a free
    intercept. And the paper permits ``alpha + beta`` slightly above 1 under
    numerical bounds where this refuses it. The paper also notes that ``u``
    is a **downward biased** estimate of the equicorrelation -- ``E[u]`` is
    about 0.20 for a true 0.30 at ``m = 6`` -- and offers an alternative
    ``1 - (1/(n-1)) * sum((r_i - rbar)**2)``, which this does not.

    ``blocks`` maps a name to a subset of ``features``: the model then
    estimates one number per block and one per pair of blocks, which is the
    useful middle between one correlation and all of them. Every feature
    must be in exactly one block, and a block needs at least two.

    Outputs, all read from the state **before** the row (so they are safe as
    features for that same row): ``u``, ``rho`` (the level before the row),
    ``loglik`` (the row's Gaussian log-density in standardised coordinates
    under that level) and ``n_eff``. With ``K`` named blocks the first two
    become ``u_<A>`` per block then ``u_<A>_<B>`` per pair, and ``rho_*``
    likewise, with one ``loglik`` over all of them; ``coef`` is the
    correlation values in that order. ``u`` and ``loglik`` are null until
    every feature has a positive pre-row variance.

    Not a regression: no targets, and nothing residual-based applies. Needs
    at least two features.
    """
    model: dict[str, Any] = {
        "type": "deco",
        "dynamics": dynamics,
        "alpha": alpha,
        "beta": beta,
        "blocks": [[k, list(v)] for k, v in blocks.items()] if blocks else None,
    }
    return _common(name, model, targets=[features[0]], features=features, **common)


@_checked
def bocpd(
    name: str,
    *,
    features: list[str],
    hazard: float = 250.0,
    hazard_col: str | None = None,
    emission: str = "diag",
    prior_mean: list[float] | None = None,
    prior_kappa: float | None = None,
    prior_nu: float | None = None,
    prior_scale: list[float] | None = None,
    robust_beta: float | None = None,
    prune_below: float | None = None,
    max_run: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Bayesian online changepoint detection (Adams & MacKay 2007;
    ENHANCEMENTS E61, Task 55).

    Every other detector here answers "has something changed?" with a
    statistic. This one keeps a **posterior over how long the current run
    has lasted**, so the answer carries the age of the regime with it:
    "we are 40 rows into a regime" is different information from "something
    broke".

    Their Algorithm 1, in log space, with ``H = 1/hazard``::

        growth:      P(r_t = r+1, x_1:t) = P(r_t-1 = r, x_1:t-1) pi_r (1 - H)
        changepoint: P(r_t = 0,   x_1:t) = sum_r P(r_t-1 = r, x_1:t-1) pi_r H

    where ``pi_r`` is run ``r``'s posterior predictive for this row. Slot
    ``r`` keeps the conjugate statistics of exactly the ``r`` rows that
    hypothesis says preceded this one in the run -- so slot 0 holds none and
    its predictive is the prior's, which is what makes "a new run starts
    here" something the data can vote on. A row costs ``O(runs * d^2)``, and
    the run vector would grow by one every row, so runs below ``prune_below``
    of the mass are dropped and ``max_run`` folds every longer run into the
    last kept one -- which takes their mass and keeps its own statistics, so
    it bounds how much history *any* run holds, not only the length of the
    vector, and ``run_mode`` saturates one below it.

    Outputs ``p_change``, ``run_mode`` and ``run_mean``, ``pred_<f>`` (the
    pre-row predictive mean), ``logscore`` (the row's log predictive
    density) and ``n_eff``. ``halflife``/``lam`` are refused: the run-length
    posterior is what forgets and ``hazard`` is how fast.

    **``run_mode`` is the answer and ``p_change`` is the alarm**, and they
    are not the same quality of signal. ``run_mode`` is the run length
    before the row, so ``t - run_mode`` is the row the current run began on.
    ``p_change`` is a per-row likelihood ratio: spiky, and as big as the
    break is against the prior scale. Measured -- a tenfold variance step
    takes it to 0.83 on the row itself; a four-sigma mean shift under a
    diffuse prior barely lifts it; a change in correlation alone never moves
    it. The run length finds all three within a few rows and dates them to
    the right row.

    It is ``P(r_t <= 1)`` and not ``P(r_t = 0)`` because the changepoint
    branch and the growth branch share the same predictive, which makes the
    normalised mass at ``r = 0`` *exactly* ``H`` on every row whatever the
    data. Row one of a group reports nothing: ``P(r <= 1)`` is 1 there
    however the row looks.

    ``emission="diag"`` (the default) is a normal-inverse-gamma per feature
    and ``"gaussian"`` a normal-inverse-Wishart over all of them; both are
    exact conjugate updates, and the second is the one that can see a break
    in the *correlation* with the marginals unchanged. ``prior_scale`` is
    the prior guess at the variance and is the one parameter to set from
    your data: too large and the model goes quiet, because no row is ever
    surprising under a predictive that wide and a real break is never found.
    ``prior_nu`` and ``prior_scale`` are ``2a`` and ``2b`` in the gamma
    parametrisation, which is how Adams and MacKay give their own finance
    example (``a = 1``, ``b = 1e-4``, ``hazard = 250``).

    ``emission="robust"`` weights each row by ``(pi(x)/pi(mode))**robust_beta``
    -- in what the run learns *and* in the message it passes on, so a
    20-sigma row is atypical under every run, every tempered likelihood is
    about 1, and nothing moves. Without it that one row *is* a changepoint
    (``p_change`` 0.91) and the run it starts carries the outlier in its
    mean. The knob is a trade: a whole new regime is a run of individually
    forgiven rows, so above about ``robust_beta = 0.2`` nothing is ever
    detected again. The default, 0.1, ignores the outlier and still dates a
    four-sigma shift to the right row.

    ``hazard_col`` reads the hazard per row from a column (declared in the
    targets slot, the way a weight is) instead of one number. A null or
    non-finite value there falls back to ``hazard``; a value of 1 or less is
    an error naming the row, since a hazard is the expected rows between
    changepoints. :meth:`ModelBank.predict` reads the column too, so it
    gives the same answer the step would for that row.

    ``prior_scale`` is the prior scale of the variance: a positive number,
    or a symmetric positive-definite ``d x d`` matrix (default: the
    identity). ``prior_mean`` is ``mu_0`` (default: zeros), ``prior_kappa``
    the weight of that mean in rows (default 1.0), and ``prior_nu`` the
    degrees of freedom (default ``d + 2``, the smallest that gives the
    Wishart a mean). ``min_periods`` gates
    what is reported, never what is learned. A row whose predictive cannot
    be evaluated reports nulls, leaves the posterior where it stands, and is
    counted in :meth:`ModelBank.solve_failures`.
    """
    model: dict[str, Any] = {
        "type": "bocpd",
        "hazard": hazard,
        "hazard_col": hazard_col,
        "emission": emission,
        "prior_mean": prior_mean,
        "prior_kappa": prior_kappa,
        "prior_nu": prior_nu,
        "prior_scale": prior_scale,
        "robust_beta": robust_beta,
        "prune_below": prune_below,
        "max_run": max_run,
    }
    targets = [hazard_col] if hazard_col is not None else [features[0]]
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def corrchange(
    name: str,
    *,
    features: list[str],
    kind: str = "monitor",
    span_rows: int | None = None,
    alpha: float = 0.05,
    alpha_adjust: str = "bonferroni",
    bandwidth: int | None = None,
    scalar: bool = False,
    crit: float | None = None,
    n_perm: int | None = None,
    permute_every: int | None = None,
    perm_block: int | None = None,
    norm: str = "l1",
    seed: int | None = None,
    reset: bool = False,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Has the correlation structure changed? (ENHANCEMENTS E59, Task 54)

    Two tests, because there are two questions.

    ``kind="monitor"`` is the **closed-sample** constancy test of Wied,
    Krämer & Dehling (2012), run over consecutive spans of ``span_rows``
    rows. At the last row of a span, per pair::

        Q = max_{2<=j<=T} (j/sqrt(T)) * |rho_j - rho_T| / D

    with ``rho_j`` the sample correlation of the span's first ``j`` rows and
    ``D`` the delta-method long-run standard deviation of ``rho``: the five
    raw moments ``(x^2, y^2, x, y, xy)`` centred at their span means, their
    Bartlett long-run covariance at bandwidth ``floor(ln T)``, mapped to
    ``(var_x, var_y, cov)`` and then to ``rho``. Under the null ``Q``
    converges to ``sup|B|``, a Brownian bridge, so the critical value is the
    Kolmogorov quantile -- **computed** from the series, not pinned, and it
    reproduces the published 1.3581 at 5%. Over the pairs the statistic is
    the maximum and the level is ``alpha / npairs`` (``alpha_adjust``).

    The paper's own sequential form, with a boundary function, is Wied &
    Galeano (2013), which nobody here has read; the closed test run span by
    span is what ships. The cost is a delay of at most ``span_rows`` rows and
    the benefit is a null with published tables — the size and power in
    their Tables 1 and 2 are what `tests/test_corrchange.py` holds it to.

    ``scalar=True`` runs the same CUSUM on the **equicorrelation** of the
    standardised row (`deco`'s ``u``) instead of every pair, which is one
    statistic however many columns there are. ``halflife``/``lam``
    parametrise that standardiser and are accepted only there; neither kind
    decays anything else, so they are refused otherwise.

    ``kind="window"`` is ``norm(vech(R_pre - R_post))`` over two adjacent
    blocks of ``span_rows`` rows -- how *big* the change is, rather than
    whether the span was constant. ``crit`` is a fixed threshold; without
    one the critical value is a **permutation** quantile: ``n_perm`` draws
    of the pooled rows shuffled between the two windows, in blocks of
    ``perm_block`` so that serial dependence does not make the null too
    liberal, redrawn every ``permute_every`` rows.

    It is not a sign-flip null, which is what a first reading of the
    literature suggests: negating a whole row leaves every ``x x'`` and so
    every correlation matrix exactly where it was, so a sign-flip null has
    no spread at all.

    Outputs ``stat``, ``crit``, ``flag`` and ``since_flag`` (learned rows
    since the last flag), plus ``n_eff``; all null except where a statistic
    is due. ``reset=True`` empties the windows at a flag (``"window"``
    only -- ``"monitor"``'s spans are disjoint already).

    ``span_rows`` is the rows per comparison block and is required by both
    kinds: at least 8 for ``"monitor"``, at least 3 for ``"window"``.
    ``bandwidth`` overrides the Bartlett bandwidth ``floor(ln T)``. ``norm``
    is ``"l1"`` (the default) or ``"linf"``.
    ``n_perm`` defaults to 200, ``permute_every`` to 50 and ``perm_block``
    to 1; ``seed`` (default 0) seeds the permutation draws, so two runs with
    the same seed report the same critical values.

    A row is always part of the span reported **on** it -- the report comes
    before the update, which is what makes the flag out of sample -- so a
    zero-weight row is reported as if it would be learned, and then does not
    enter the span, does not advance ``since_flag`` for the rows after it,
    and does not reset it if it flags.
    """
    model: dict[str, Any] = {
        "type": "corrchange",
        "kind": kind,
        "span_rows": span_rows,
        "alpha": alpha,
        "alpha_adjust": alpha_adjust,
        "bandwidth": bandwidth,
        "scalar": scalar,
        "crit": crit,
        "n_perm": n_perm,
        "permute_every": permute_every,
        "perm_block": perm_block,
        "norm": norm,
        "seed": seed,
        "reset": reset,
    }
    return _common(name, model, targets=[features[0]], features=features, **common)


@_checked
def hmm(
    name: str,
    *,
    features: list[str],
    k: int,
    precision_prior: float,
    covariance: str = "full",
    learn: bool = True,
    transition_prior: float | None = None,
    transition: list[float] | None = None,
    means: list[float] | None = None,
    covs: list[float] | None = None,
    warm_rows: int | None = None,
    seed_rule: str | None = None,
    seed: int | None = None,
    exog_tvtp: str | None = None,
    tvtp_coef: list[list[float]] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A Gaussian hidden Markov model, filtered online (ENHANCEMENTS E60,
    Task 53).

    ``ew_class`` classifies a row against *labelled* Gaussians. This does
    the same arithmetic with no labels: the state is hidden, and a
    transition matrix carries information from one row to the next. That is
    the difference between "which regime does this row look like" and
    "which regime are we in", and the second is usually the question.

    Hamilton's filter, one row at a time. Before the row, from the filtered
    ``p`` the previous row left (uniform before the first)::

        p1_l   = sum_k p_k * Pi_kl                     the predicted state
        f_l    = N(x | mu_l, Sigma_l + r_l I)          the state's density
        loglik = log sum_l p1_l f_l                    the row's surprise
        p_l   <- p1_l f_l / sum                        the filtered state

    Everything reported is read **before** the row is learned from, so an
    ``hmm`` output is safe as a feature for that same row. The densities go
    through the same path ``ew_class`` uses, with the same decaying
    ``precision_prior`` ridge -- **required** here as it is there, because a
    state's centred co-moments start at zero and a zero matrix has no
    density.

    Each state's accumulator then takes the row at weight ``w * p_l``. The
    responsibilities sum to ``w``, so ``n_eff`` is the shared recursion
    untouched. The transition matrix is learned from the **filtered joint of
    consecutive states**::

        xi_kl = p_k(t-1) Pi_kl f_l / sum over all pairs
        A_kl <- decay * A_kl + w * xi_kl
        Pi_kl = (A_kl + tau_kl) / sum_l (A_kl + tau_kl)

    with ``tau`` a Dirichlet pseudo-count per cell (``transition_prior``,
    default 1), which is what keeps a never-visited row of ``Pi`` a
    distribution. Giving ``transition`` spreads that mass over the given
    matrix instead of flat, so the given matrix is the prior mean.

    ``means`` and ``covs`` (``K x d`` and ``K`` matrices of ``d x d``, both
    flattened row-major) give the states outright and there is no warm-up.
    Each ``covs`` block must be symmetric and positive definite -- a state
    with no density takes no responsibility for any row -- and the pair
    enters at **one row's weight**, so under ``learn=True`` the stream
    washes the given states out at the ordinary rate and under
    ``learn=False`` they are held exactly. Otherwise the first ``warm_rows``
    learned rows are buffered, ``kmeans``' ``seed_rule`` chooses centres
    among them, and the buffer is replayed through those centres as hard
    assignments -- every output is null until then, as ``kmeans``' are.
    ``learn=False`` with no states given is refused: there would be nothing
    to filter with.

    ``min_periods`` gates what is **reported**, never what is learned: a row
    below it moves the filter and shows nulls, as it does in every other
    model here.

    ``exog_tvtp`` names a column (declared like ``weight``, not a feature)
    whose value drives the matrix instead: ``Pi_kl(t) = softmax_l(A_kl +
    B_kl z_t)`` from the fixed ``tvtp_coef = [A, B]``. The count-based
    learning is off under it; ``A`` and ``B`` are fitted elsewhere. A row
    whose ``exog_tvtp`` is null or non-finite is filtered with ``z = 0``,
    the base transition, and is otherwise an ordinary row.
    :meth:`ModelBank.predict` reads the column too, so it gives the same
    answer the step would for that row.

    Outputs ``p_<k>``, ``p1_<k>``, ``state``, ``loglik`` and ``n_eff``, with
    the state means as ``coef``.

    ``covariance`` is ``"full"`` (the default), ``"shared"`` or
    ``"diagonal"``, as for :func:`ew_class`. ``warm_rows`` defaults to 50;
    ``seed_rule`` is ``"lloyd"`` (the default), ``"first"``, ``"farthest"``
    or ``"kmeanspp"``, and ``seed`` (default 0) seeds it, so the warm-up is
    reproducible.

    **A limitation worth knowing.** A single extreme row can be captured by
    one state, moving its mean far from the data; in mean form a state with
    zero responsibility keeps its moments, so a state that stops winning
    never forgets, and the mixture is left short one state. A larger
    ``precision_prior``, filtering with given states (``learn=False``), or
    cleaning the input upstream are the mitigations.

    **A regime that lives only in the covariance needs the covariances to
    start from.** The default seeding is k-means over the rows, and two
    zero-mean states differ in nothing k-means can see, so it splits them by
    *direction* and the filter never recovers: measured at 56 % accuracy on
    a two-state stream, against 98 % for the same filter given ``covs``.
    Pass ``means`` and ``covs``, or give it a feature in which the regime is
    a shift in location. And keep ``precision_prior`` small against the
    data's scale -- a ridge of 1.0 on a covariance whose entries are about
    1.0 halves every correlation, which in that stream is the whole of the
    signal. Both are measured in ``docs/REGIMES.md`` §1.
    """
    model: dict[str, Any] = {
        "type": "hmm",
        "k": k,
        "covariance": covariance,
        "precision_prior": precision_prior,
        "learn": learn,
        "transition_prior": transition_prior,
        "transition": transition,
        "means": means,
        "covs": covs,
        "warm_rows": warm_rows,
        "seed_rule": seed_rule,
        "seed": seed,
        "exog_tvtp": exog_tvtp,
        "tvtp_coef": tvtp_coef,
    }
    targets = [exog_tvtp] if exog_tvtp is not None else [features[0]]
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def rcov(
    name: str,
    *,
    features: list[str],
    kind: str = "kernel",
    kernel: str = "parzen",
    bandwidth: int | None = None,
    jitter: int = 2,
    theta: float = 1.0,
    psd: bool = True,
    block_rows: int | None = None,
    max_bandwidth: int | None = None,
    preavg_rows: int | None = None,
    noise_stride: int | None = None,
    iv_stride: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A block's realised covariance, robust to microstructure noise
    (ENHANCEMENTS E57, Task 50).

    A plain realised covariance over ticks is biased by noise -- each price
    is the efficient one plus an error, and the error's variance accumulates
    with every tick -- and attenuated by asynchrony. Both are estimated away
    by published estimators that are **sums over lags**, which is exactly
    what a stream can accumulate.

    Rows are **returns**: difference upstream (`.diff().over(by)` after
    :func:`polars_online.prep.refresh_time`). There is no decay and no
    per-row output but ``n_eff``, because the value is the block: ``rcov``
    requires ``group`` and ``group_close``, and the estimate rides in the
    row that close emits (:meth:`ModelBank.closed_groups`).

    ``kind="plain"`` is ``sum(x x')``, which at close equals ``n`` times an
    ``ew_cov(lam=1)``'s uncentred second moment to the bit -- the
    cross-check, and the reference the other two are measured against.

    ``kind="kernel"`` (the default) is Barndorff-Nielsen, Hansen, Lunde &
    Shephard's multivariate realised kernel::

        K = sum_{h=-H}^{H} k(h/(H+1)) * Gamma_h
        Gamma_h = sum_j x_j x'_{j-h},  Gamma_{-h} = Gamma_h'

    with the Parzen kernel -- a positive-definite function, so ``K`` is PSD
    up to rounding, and 0.97 efficient against the quadratic spectral's
    0.93; the Bartlett kernel is not consistent here and is not offered. The
    end points are jittered by averaging the first and last ``jitter``
    observations; ``1`` is no jitter, and the paper's own ``m = 1..4`` move
    the estimate by under 0.5 %, so the default of 2 is immaterial.
    ``bandwidth`` is a fixed ``H``; left out it is their
    ``H = ceil(c* xi^{4/5} n^{3/5})`` with ``c* = 3.5134``, which needs
    ``block_rows`` -- the ring has to be sized before the first row and ``n`` is
    known only at the close. ``block_rows`` is a sizing hint, not a limit: a
    longer block runs, clipped, and reports ``bandwidth_used``. ``max_bandwidth``
    fixes the ring depth itself (default ``ceil(c* block_rows^{3/5})`` under the
    automatic bandwidth, the depth at which the noise equals the block's
    integrated variance); it must not cap the ring below a fixed
    ``bandwidth``. ``kernel`` takes only ``"parzen"``.

    ``kind="preavg"`` is Christensen, Kinnebrock & Podolskij's modulated
    realised covariance: the returns are pre-averaged over ``k_n =
    floor(theta * sqrt(block_rows))`` with ``g(x) = min(x, 1-x)``, which averages
    the noise away, and the residual bias is subtracted. ``psd=False`` is
    that balanced, bias-corrected form (optimal rate, not guaranteed PSD);
    ``psd=True`` (the default) is the longer window ``k_n =
    ceil(theta * block_rows^0.6)`` without the bias term, and clips any negative
    eigenvalue, reporting ``psd_repaired``. ``preavg_rows`` fixes ``k_n``
    (at least 2) instead of deriving it from ``block_rows``, for ``"preavg"``
    only.

    ``noise_stride`` (default 1) and ``iv_stride`` (default 20) are the two
    subsampled grids behind an automatic bandwidth: the noise variance
    ``omega2`` from the dense one, deliberately biased upward as BNHLS
    accept, and the integrated variance ``iv_sparse`` from the sparse one,
    each averaged over the stride's offsets.

    The closed row carries ``rcov`` and ``rcorr`` (``vech`` of the upper
    triangle), ``rcov_n``, ``rcov_kind``, ``bandwidth_used``, ``omega2``,
    ``iv_sparse``, ``iq`` (a realised-quarticity proxy, and labelled one)
    and ``psd_repaired``; all null for a block too short to estimate from.

    ``weight`` is taken only as 0 or 1 -- a sum over returns has no
    fractional row -- and a zero-weight row advances the clock and enters no
    ring. ``halflife``/``lam`` are refused: the block boundary is
    ``group_close``'s, not a decay's.

    A gap over ``max_dclock``, or a session change, splits the block into
    **stretches**: the returns on either side of it are not adjacent, and a
    covariance of adjacent returns is the whole statistic. Each stretch is
    closed as the last one is -- leading jitter, interior, trailing jitter --
    and the lagged sums add over stretches, so no product pairs two returns
    across the break. A stretch too short to reach its trailing jitter
    contributes only what it had already emitted, and ``rcov_n`` says how
    many effective returns there were in total.
    """
    model: dict[str, Any] = {
        "type": "rcov",
        "kind": kind,
        "kernel": kernel,
        "bandwidth": bandwidth,
        "jitter": jitter,
        "theta": theta,
        "psd": psd,
        "block_rows": block_rows,
        "max_bandwidth": max_bandwidth,
        "preavg_rows": preavg_rows,
        "noise_stride": noise_stride,
        "iv_stride": iv_stride,
    }
    return _common(name, model, targets=[features[0]], features=features, **common)


#: The model types with no target column: their outputs are read from the
#: state before each row, their ``targets`` mirror ``features[0]`` for the
#: plumbing, and nothing residual-based applies to them. ``ew_class`` is
#: not one -- its label column travels as the target -- though it predicts
#: no number either, and refuses the residual switches the same way.
UNSUPERVISED = frozenset(
    {"ew_cov", "kmeans", "micro", "deco", "rcov", "hmm", "corrchange", "bocpd"}
)

_NUMERIC_KEYS = _numeric_keys()
