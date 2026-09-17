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
    label_delay: float | None


class CommonKwargs(ExprKwargs, total=False):
    """What the builders take as ``**common``: the above plus the group and
    its close policy."""

    group: str | None
    group_close: str | None
