"""The trend model's builder, :func:`holt`: a part of
:mod:`polars_online._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`polars_online.spec` documents it with the rest.
"""

from __future__ import annotations

from typing import Any, Unpack

from polars_online._duration import Duration
from polars_online._kwargs import CommonKwargs
from polars_online._spec import TargetList, _checked, _common


@_checked
def holt(
    name: str,
    *,
    targets: TargetList,
    trend_half_life: float | Duration | None = None,
    trend: bool = True,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Holt's linear trend: the target's own level and slope, extrapolated.

    It takes no features, as only :func:`seqtest` among the other models
    does -- the forecasting baseline a feature-based model should have to
    beat. If a regression cannot
    outperform "the series is going up at about this rate", its features are
    not earning their place. Run it in the same bank and compare the two
    ``sigma``, or let a :func:`seqtest` with ``a`` and ``b`` say which
    predicts closer.

    .. rubric:: The fit

    Per row and target, with ``s`` the clock since the target was last
    observed: this row's delta included, so ``s`` is the row's own delta on
    a stream with no gaps. ``w`` is the row's weight, and ``W``, ``V`` the
    weight the level and the trend have gathered, each decayed on its own
    half-life (``lam_l = 0.5 ** (s / half_life)``, the spec's own, and
    ``lam_b`` likewise at ``trend_half_life``):

    .. code-block:: text

        pred = l + b * s                                   extrapolate s clock units ahead
        l'   = (lam_l * W * pred + w * y) / (lam_l * W + w)
        b'   = (lam_b * V * b + w * (l' - l) / s) / (lam_b * V + w)

    A target's first observation sets ``l = y`` and ``W = w``, and predicts
    nothing; the trend starts at 0. Level and trend are weighted means, as
    every accumulator here is: a row at weight ``w`` counts ``w`` times, and an
    infinite half-life forgets nothing and fits the whole history. The gains
    ``w / (lam * W + w)`` start at 1 and fall to the textbook's fixed
    ``1 - lam`` as the weight saturates. So a new series is followed sooner,
    and the fit is statsmodels' ``Holt`` from there. A row at the last row's
    clock is a second observation the level takes in; the trend holds, since a
    move over no clock has no slope. A row with a null target or a zero weight
    leaves ``l`` and ``b`` where the last observation put them and carries its
    clock to the next one. So it gives the same numbers as if it were absent.
    The trend is per clock unit, so an irregular clock extrapolates the right
    distance.

    .. rubric:: Parameters

    The level forgets at the spec's ``half_life`` (or ``lam``), in clock units,
    ``inf`` included: one knob under one name, as for every model.
    ``level_half_life`` is refused, with a message naming ``half_life``.

    ``trend_half_life``
        How fast the trend forgets, in clock units, above 0. Default four times
        the level's half-life; ``inf`` is the whole history's drift, not a
        trend pinned at zero.
    ``trend``
        ``False`` fits the level alone: the trend is held at zero and the
        forecast is flat, which is simple exponential smoothing. The level is
        then the weighted mean of the observations, each at its weight times
        ``0.5 ** (age / half_life)``, which is pandas'
        ``ewm(adjust=True)`` for unit weights. ``trend_half_life`` is refused
        beside it. Default ``True``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    There is no seasonal term: a seasonal index is a ``group`` on the phase,
    which the bank already does.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#holt
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#holt>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not
        null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
    ``coef``
        ``[level, trend]`` per target, the whole state; :func:`coef_index`
        names the two. Null for a target not yet observed, which has no
        level to report. The trend is 0 under ``trend=False``.

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. :meth:`polars_online.ModelBank.predict` extrapolates
    over the clock distance from the row the model last learned, capped by
    ``gap_cap``.

    .. rubric:: Example

    .. code-block:: python

        baseline = po.spec.holt(
            "baseline", targets=["y"], clock="t", gap_cap=600.0,
            half_life=200.0,          # how fast the level forgets
            trend_half_life=2000.0,   # how fast the trend forgets; inf is the whole history's drift
        )
        out = po.ModelBank([baseline]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``features``, which this model has not got; and ``ValueError`` for a
    ``trend_half_life`` not above 0, and ``trend_half_life`` beside
    ``trend=False``.
    """
    model: dict[str, Any] = {
        "type": "holt",
        "trend_half_life": trend_half_life,
    }
    if not trend:
        # Absent when on, as the Rust spec leaves it out, so a bank reports
        # the dict this made (`ModelBank.specs`).
        model["trend"] = False
    return _common(name, model, targets=targets, features=[], **common)
