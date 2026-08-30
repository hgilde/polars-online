"""Spec builders: plain dicts, validated eagerly by the Rust core."""

from __future__ import annotations

import json
import math
from typing import Any

from polars_online._polars_online import spec_output_fields, validate_spec


def _json(spec: dict[str, Any]) -> str:
    """JSON has no infinity literal, but ``halflife=inf`` is meaningful (it pins
    a coefficient), so non-finite floats are encoded as strings the Rust side
    understands."""

    def enc(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return "inf" if v > 0 else ("-inf" if v < 0 else "nan")
        if isinstance(v, dict):
            return {k: enc(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [enc(x) for x in v]
        return v

    return json.dumps(enc(spec))


__all__ = ["ewridge", "huber", "kalman", "lasso", "output_fields", "quantile", "rls"]


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
    min_periods: float | None = None,
    coef_every: int = 0,
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
        "group": group,
    }
    validate_spec(_json(spec))
    return spec


def ewridge(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    ridge: float | list[float] | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    standardize: bool = False,
    ridge_decay: bool = False,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    **common: Any,
) -> dict[str, Any]:
    """EW-ridge spec (docs/PLAN.md §4.1).

    Math: EW means ``S = EW[x x^T]``, ``r_j = EW[x y_j]`` with per-row decay
    ``0.5 ** (d_clock / halflife)``; coefficients solve
    ``(S + ridge * D) beta_j = r_j`` (D = identity minus the intercept slot) on a
    schedule (``solve_every`` clock units, default halflife/50). Predictions use
    the last solved coefficients and the state *before* the row's update.
    """
    model: dict[str, Any] = {
        "type": "ew_ridge",
        "ridge": ridge,
        "feature_sets": list(feature_sets.items()) if feature_sets else None,
        "standardize": standardize,
        "ridge_decay": ridge_decay,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
    }
    return _common(name, model, targets=targets, features=features, **common)


def output_fields(spec: dict[str, Any]) -> list[str]:
    """Struct field names this spec will produce, in order."""
    return spec_output_fields(_json(spec))


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

    Math: maintains ``P = A^-1`` with ``A = sum of decayed w z z^T + ridge I``
    via Sherman-Morrison, so coefficients update every row with no solve
    staleness: ``g = P z / (1/w + z' P z)``,
    ``beta_j += g (y_j - z' beta_j)``, ``P -= g (P z)'``. ``ridge`` sets
    ``P0 = I / ridge`` and (unlike ew_ridge) penalizes the intercept.

    Null policy deviation: a row with ANY null target is predict-only for all
    targets, because P is shared across targets.
    """
    model: dict[str, Any] = {"type": "rls", "ridge": ridge, "coef0": coef0}
    return _common(name, model, targets=targets, features=features, **common)


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
    }
    return _common(name, model, targets=targets, features=features, **common)


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
