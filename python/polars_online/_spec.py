"""Spec builders: plain dicts, validated eagerly by the Rust core."""

from __future__ import annotations

import json
from typing import Any

from polars_online._polars_online import spec_output_fields, validate_spec

__all__ = ["ewridge", "output_fields", "rls"]


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
