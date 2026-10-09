"""The shared keyword parameters of every spec, as ``TypedDict`` classes (PEP 692).

The builders in ``_spec.py`` take the shared parameters as ``**common``; with
``**kwargs: Any`` an editor shows nothing and a typo is found at runtime.
Annotating them as ``Unpack[...]`` of the classes below gives completion and
type checking without changing a call (docs/IMPROVEMENTS.md U4).

``CommonKwargs`` is pinned to the shared parameters by
``tests/test_kwargs_typing.py``: same keys, same annotations, same required
set. Change a shared parameter and the test says so. Defaults are not repeated
here -- a TypedDict has none; the builder's signature is where they live.

There were once 21 further classes here, one per model, mirroring each
builder's *own* parameters to type the expression namespace's ``**kwargs``.
Task 85 removed that namespace, and the test that held each class to its
builder went with it. What was left was 21 unconsumed copies of builder
signatures with nothing holding them to the originals -- which is exactly the
drift the paragraph above exists to prevent. They were deleted on 2026-09-17
rather than left to rot; rebuild them from the builders if a typed keyword
surface is ever wanted again.

No ``from __future__ import annotations`` here: under it ``Required[...]`` is
a string the TypedDict machinery does not look inside, so every key would be
optional at runtime and the test below could not see the required ones.
"""

from typing import TypedDict

from polars_online._duration import Duration

__all__ = [
    "CommonKwargs",
    "ExprKwargs",
]


class ExprKwargs(TypedDict, total=False):
    """The parameters every model shares, minus ``group`` and ``group_close``.

    The base :class:`CommonKwargs` inherits from. The name is historical: it
    was the set the expression namespace took, before that namespace was
    removed. It survives as the shared base, and
    ``tests/test_kwargs_typing.py`` holds it to the shared parameters.
    """

    fit_intercept: bool
    clock: str | None
    half_life: float | Duration | list[float | Duration] | None
    lam: float | None
    gap_cap: float | Duration | None
    restart_after_step_back: float | Duration | None
    session: str | None
    session_gap: float | Duration | None
    weight: str | None
    min_weight: float | list[float] | None
    min_settled_frac: float | None
    max_error_inflation: float | None
    emit_error_inflation: bool
    emit_se_coef: bool
    emit_clocks: bool
    coef_every: float | Duration | None
    max_rows_between_coefs: int | None
    emit_sigma: bool
    emit_zscore: bool
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
    drift_threshold: float | Duration | None
    drift_action: str
    emit_calibration: bool
    calibration_half_life: float | Duration | None
    emit_breaks: bool
    breaks_half_life: float | Duration | None
    emit_robust_se: bool
    robust_se_half_life: float | Duration | None
    robust_se_lags: int | None
    emit_specification: bool
    specification_half_life: float | Duration | None
    ljung_box_lags: int | None
    horizon_rows: int | None
    emit_tails: bool
    tails_half_life: float | Duration | None
    emit_influence: bool
    influence_half_life: float | Duration | None
    emit_feature_health: bool
    feature_health_half_life: float | Duration | None
    embargo: float | Duration | None


class CommonKwargs(ExprKwargs, total=False):
    """What the builders take as ``**common``: the above plus the group and
    its close policy."""

    group: str | None
    group_close: str | None
