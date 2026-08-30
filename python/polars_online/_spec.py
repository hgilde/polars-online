"""Spec builders: plain dicts, validated eagerly by the Rust core."""

from __future__ import annotations

import json
from typing import Any

from polars_online._polars_online import spec_output_fields, validate_spec

__all__ = ["ewridge", "lasso", "output_fields", "rls"]


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
    validate_spec(json.dumps(spec))
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
    return spec_output_fields(json.dumps(spec))


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
