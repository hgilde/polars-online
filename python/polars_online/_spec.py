"""Spec builders: plain dicts, validated eagerly by the Rust core."""

from __future__ import annotations

import functools
import json
import math
import numbers
import types
import typing
from collections.abc import Callable
from typing import Any

import polars as pl

from polars_online._polars_online import spec_output_fields, spec_output_index, validate_spec


def _json(spec: dict[str, Any]) -> str:
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
                raise ValueError(f"spec {json.dumps(spec.get('name'))}: {key} must not be NaN")
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
    "kalman": frozenset({"coef_halflife", "q"}),
    "sgd": frozenset({"clip_gradient"}),
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
            if _matches(value, int) and int in (hint, *typing.get_args(hint)) and value < 0:
                raise ValueError(f"{who}: {key} must be >= 0, got {value}")
            if key not in inf_ok and not _finite(value):
                raise ValueError(f"{who}: {key} must be finite, got {_got(value)}")
        return fn(*args, **kwargs)

    return wrapper


__all__ = [
    "ew_cov",
    "ewridge",
    "ftrl",
    "holt",
    "huber",
    "kalman",
    "lasso",
    "output_fields",
    "pa",
    "quantile",
    "rls",
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
    resid_quantiles: list[float] | None = None,
    emit_autocorr: bool = False,
    resid_autocorr_lag: int | None = None,
    emit_drift: bool = False,
    drift_delta: float | None = None,
    drift_threshold: float | None = None,
    drift_action: str = "flag",
    group: str | None = None,
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
        "resid_quantiles": resid_quantiles,
        "emit_autocorr": emit_autocorr,
        "resid_autocorr_lag": resid_autocorr_lag,
        "emit_drift": emit_drift,
        "drift_delta": drift_delta,
        "drift_threshold": drift_threshold,
        "drift_action": drift_action,
        "group": group,
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
    coef0: list[list[float]] | None = None,
    session_shrink: float | None = None,
    long_halflife: float | None = None,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    **common: Any,
) -> dict[str, Any]:
    """EW-ridge spec (docs/PLAN.md §4.1).

        ``coef0`` shrinks toward a stated belief instead of toward zero, one vector
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

        Math: EW means ``S = EW[x x^T]``, ``r_j = EW[x y_j]`` with per-row decay
        ``0.5 ** (d_clock / halflife)``; coefficients solve
        ``(S + ridge * D) beta_j = r_j`` (D = identity minus the intercept slot) on a
        schedule (``solve_every`` clock units, default halflife/50). Predictions use
        the last solved coefficients and the state *before* the row's update.
    """
    model: dict[str, Any] = {
        "type": "ew_ridge",
        "ridge": ridge,
        "feature_sets": [[k, list(v)] for k, v in feature_sets.items()] if feature_sets else None,
        "standardize": standardize,
        "ridge_decay": ridge_decay,
        "coef0": coef0,
        "session_shrink": session_shrink,
        "long_halflife": long_halflife,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
    }
    return _common(name, model, targets=targets, features=features, **common)


def _numeric_keys() -> frozenset[str]:
    """Every parameter, across the builders, whose annotation admits a float."""
    keys = set()
    for fn in (_common, ewridge, rls, lasso, kalman, huber, quantile, ftrl, ew_cov, sgd, pa, holt):
        for key, hint in typing.get_type_hints(getattr(fn, "__wrapped__", fn)).items():
            leaves = {hint, *typing.get_args(hint)}
            leaves |= {a for h in list(leaves) for a in typing.get_args(h)}
            leaves |= {a for h in list(leaves) for a in typing.get_args(h)}
            if float in leaves:
                keys.add(key)
    return frozenset(keys)


def output_fields(spec: dict[str, Any]) -> list[str]:
    """Struct field names this spec will produce, in order."""
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
    never drift from the strings.
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


def coef_index(spec: dict[str, Any]) -> pl.DataFrame:
    """The layout of each ``coef`` list, one row per position.

    ``coef`` is flat: (target x combo) slots, each contributing its terms in
    order. This maps ``position`` -> (``target``, combo metadata, ``term``),
    where ``term`` is ``"intercept"``, a feature name, or -- for ``holt`` --
    ``"level"`` / ``"trend"``::

        ci = po.spec.coef_index(spec)
        pos = ci.filter(
            (pl.col("target") == "y") & (pl.col("ridge") == 0.5)
            & (pl.col("term") == "x1")
        )["position"].item()
        slope = out["m"].struct.field("coef@h100").list.get(pos)

    Derived from :func:`output_index` (the slot order comes from the same Rust
    code that renders the names), never from parsing strings.
    """
    idx = output_index(spec)
    model = spec.get("model", {}).get("type")
    if model == "ew_cov":
        msg = "ew_cov emits statistics, not coefficients"
        raise ValueError(msg)
    if model == "holt":
        terms = ["level", "trend"]
    else:
        terms = (["intercept"] if spec.get("add_intercept", True) else []) + list(
            spec.get("features", [])
        )
    # One instance's slot order, exactly as the pred fields declare it.
    one = idx.filter(pl.col("kind") == "pred")
    first = one.row(0, named=True)
    slots = one.filter(
        (pl.col("halflife").eq_missing(first["halflife"]))
        & (pl.col("lam").eq_missing(first["lam"]))
    )
    rows = []
    pos = 0
    for slot in slots.iter_rows(named=True):
        for term in terms:
            rows.append(
                {
                    "position": pos,
                    "target": slot["target"],
                    "ridge": slot["ridge"],
                    "feature_set": slot["feature_set"],
                    "lambda": slot["lambda"],
                    "term": term,
                }
            )
            pos += 1
    return pl.DataFrame(rows)


@_checked
def rls(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    ridge: float | None = None,
    coef0: list[list[float]] | None = None,
    **common: Any,
) -> dict[str, Any]:
    """Recursive least squares spec (docs/PLAN.md section 4.2).

    Math: decayed ridge least squares solved exactly every row,
    ``A <- lam A + w z z'``, ``b_j <- lam b_j + w y_j z``, ``beta_j = A^-1 b_j``
    with ``A0 = ridge I`` and ``b0 = ridge coef0``. The state is the Cholesky
    factor ``R`` of ``A`` and ``u_j = R^-T b_j`` (square-root / QR form): a row
    is folded in by Givens rotations and ``beta`` read off by one
    back-substitution, O(k^2) per row with no solve staleness and none of the
    covariance recursion's drift (docs/IMPROVEMENTS.md C5). ``ridge`` sets
    ``A0 = ridge I`` (``P0 = I / ridge``) and (unlike ew_ridge) penalizes the
    intercept.

    Null policy deviation: a row with ANY null target is predict-only for all
    targets, because P is shared across targets.
    """
    model: dict[str, Any] = {"type": "rls", "ridge": ridge, "coef0": coef0}
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
    max_cd_iters: int | None = None,
    cd_tol: float | None = None,
    **common: Any,
) -> dict[str, Any]:
    """Lasso path with online lambda selection (docs/PLAN.md section 4.3).

    Math: coordinate descent on the standardized centered statistics held in the
    same accumulators as ew_ridge. For each penalty ``l`` in the (decreasing)
    ``lasso_path``, with ``C`` the feature correlation matrix and ``c`` the
    feature-target correlations::

        rho_i = c_i - sum_{j != i} C_ij b_j
        b_i   = soft(rho_i, l * l1_ratio) / (C_ii + l * (1 - l1_ratio))

    warm-started along the path and across solves, then unscaled with the
    intercept recovered as ``ybar - m . beta``. ``l1_ratio < 1`` is elastic net.

    Selection is free: predictions for every path point are computed anyway, so
    ``lam_selected_<target>`` is the argmin over the path of an EW mean squared
    out-of-sample error with halflife ``select_halflife`` (default: the model
    halflife). Outputs carry one pred/resid pair per path point.
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
    standardize: bool = True,
    **common: Any,
) -> dict[str, Any]:
    """Kalman / random-walk-beta dynamic linear model (docs/PLAN.md section 4.4).

    State per target: coefficient mean ``b_j`` and covariance ``P_j``. Per row
    (clock delta ``d``, row weight ``w``)::

        P_j <- P_j + Q * d / w
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
    **common: Any,
) -> dict[str, Any]:
    """Huber regression (docs/PLAN.md section 4.5).

    IRLS reweighting on the ew_ridge update: each row's weight is scaled by the
    robust weight of its *prior* residual, so it stays out-of-sample. With
    ``d = huber_delta`` in units of the EW residual std ``s``::

        w_robust = 1            if |r| <= d * s
                 = d * s / |r|  otherwise

    The weights are per target, so ``S`` is per target here (one accumulator
    each), unlike ew_ridge which shares one. Default ``huber_delta`` is 1.5.
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
    **common: Any,
) -> dict[str, Any]:
    """Quantile regression at level ``quantile`` (docs/PLAN.md section 4.5).

    The IRLS weights of the check loss, applied to the prior residual::

        w_robust = 2 * tau       * s / max(|r|, eps * s)   if r > 0
                 = 2 * (1 - tau) * s / max(|r|, eps * s)   otherwise

    ``quantile_eps`` floors |r| (in units of the EW residual std) so a
    near-zero residual cannot produce an unbounded weight.
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
    **common: Any,
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

        n_i  <- lam * n_i ;  z_i <- lam * z_i
        b_i  = 0 if |z_i| <= l1 else
               -(z_i - sign(z_i) l1) / ((beta + sqrt(n_i)) / alpha + l2)
        p    = sigmoid(z . b)
        g_i  = (p - y) * z_i * w
        z_i += g_i - ((sqrt(n_i + g_i^2) - sqrt(n_i)) / alpha) * b_i
        n_i += g_i^2

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
    **common: Any,
) -> dict[str, Any]:
    """Exponentially weighted moments of the feature columns (docs/PLAN.md 4.7).

    Not a regression: there are no targets and no coefficients, just running
    statistics of the columns you name, decayed on the same clock as every
    model here::

        W'    = lam * W + w
        m'_i  = (lam * W * m_i + w * x_i) / W'
        S'_ij = (lam * W * S_ij + w * x_i x_j) / W'

    with ``var_i = S_ii - m_i^2``, ``cov_ij = S_ij - m_i m_j`` and
    ``corr_ij = cov_ij / sqrt(var_i var_j)``. One O(k^2) update per row, which
    replaces the O(k^2) *passes* a pure-Polars pairwise EW correlation needs.

    ``stats`` selects which to emit, from ``mean``, ``var``, ``std``, ``cov``,
    ``corr`` and ``partial_corr`` (default: mean, std, corr).
    ``partial_corr`` is the correlation between two columns *controlling for
    every other column*, read off the precision matrix as
    ``-P_ij / sqrt(P_ii P_jj)``. It needs ``precision_prior``: the precision
    matrix is ``(C + s * prior * I)^-1``, solved from the co-moments on each
    row it is read (O(k^3), only when asked for); like RLS's ``P0`` the prior
    fades as data accumulates.

    Pairwise statistics are emitted for each unordered pair ``i < j``, named
    after the columns (``corr_x0_x1``, ``pcorr_x0_x1``).

    Values are read from the state *before* each row, so an ``ew_cov`` output
    can be used as a feature for that same row without leaking it.
    """
    model: dict[str, Any] = {
        "type": "ew_cov",
        "stats": stats,
        "precision_prior": precision_prior,
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
    **common: Any,
) -> dict[str, Any]:
    """Stochastic gradient descent with pluggable losses (ENHANCEMENTS E16).

    The cheap baseline: one gradient step per row, no solves, O(k) rather than
    O(k^2). Also the only model here that takes **count targets**, via
    ``loss="poisson"`` with a log link -- none of the exact solvers cover those.

    With ``eta = z . b``, ``p = link(eta)`` and ``d = dL/d(eta)``:

    ==================== ========= =========== ==============================
    loss                 link      ``p``       ``d``
    ==================== ========= =========== ==============================
    squared              identity  ``eta``     ``p - y``
    huber                identity  ``eta``     ``clamp(p - y, +/-delta)``
    quantile             identity  ``eta``     ``1{y < p} - tau``
    epsilon_insensitive  identity  ``eta``     0 inside the tube, else sign
    poisson              log       ``exp(eta)``  ``p - y``
    logistic             sigmoid   sigmoid     ``p - y``
    ==================== ========= =========== ==============================

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
    **common: Any,
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
    """
    model: dict[str, Any] = {"type": "pa", "mode": mode, "c": c, "eps": eps}
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def holt(
    name: str,
    *,
    targets: list[str],
    level_halflife: float | None = None,
    trend_halflife: float | None = None,
    features: list[str] | None = None,
    **common: Any,
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


_NUMERIC_KEYS = _numeric_keys()
