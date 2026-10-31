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
    a stream with no gaps. ``w`` is the row's weight, ``e = y - pred`` the
    forecast error, and ``W``, ``V`` the weight the level and the trend have
    gathered, each decayed on its own half-life (``lam_l = 0.5 ** (s /
    half_life)``, the spec's own, and ``lam_b`` likewise at
    ``trend_half_life``):

    .. code-block:: text

        pred = l + b * s                          extrapolate s clock units ahead
        textbook:  g_t = w / (lam_l * W + w)      W' = lam_l * W + w
                   h_t = w / (lam_b * V + w) * g / s
                                                  V' = lam_b * V + w   (s > 0)
        line fit:  S0' = S0 + w
                   a'  = S0 * (a + s) / S0'
                   Q'  = Q + S0 * w * (a + s)**2 / S0'
                   g_f = w * (1 / S0' + a'**2 / Q')
                   h_f = w * a' / Q'
        l' = pred + g * e                         g = max(g_f, g_t)
        b' = b + h * e                            h = max(h_f, h_t)

    A target's first observation sets ``l = y``, ``W = S0 = w`` and ``a = Q =
    0``, and predicts nothing; the trend starts at 0.

    The textbook gains are weighted means', as every accumulator here is: a
    row at weight ``w`` counts ``w`` times. Where they win, ``l'`` is ``(lam_l *
    W * pred + w * y) / (lam_l * W + w)``, and ``b'`` the weighted mean of
    ``b`` and the slope the level moved at, ``(l' - l) / s``. Once the weight
    saturates they are the textbook's fixed ``1 - lam``.

    The line fit's gains are those of the weighted least-squares line through
    every observation so far, on the clock, read at the last: ``S0`` its
    weight, ``a`` the observations' weighted mean age, ``Q`` the spread of
    their ages about it, none decayed. With every observation at one clock
    (``Q = 0``) no slope is determined: ``g_f = w / S0'`` and ``h_f = 0``.

    A row takes the larger gain, on the level and on the slope. An infinite
    half-life has no textbook gain, its fixed ``1 - lam`` being 0, so without
    decay the model is the weighted line fit, to rounding (statsmodels'
    ``WLS`` agrees to 1e-15). Under a half-life the line fit's gains fall as
    ``1/n`` and ``1/n**2`` and the textbook's settle, so after a warm-up of a
    few half-lives every row takes the textbook's, and statsmodels' ``Holt``
    agrees from there. The warm-up follows a new series sooner than the
    textbook would. Without the line fit, gains of ``1/n`` on both level and
    trend pinned the trend to the first observation, and the level carried
    that observation's error times ``ln n``: 11 off a flat series after
    40,000 rows without decay, at a noise of 0.5.

    A row at the last row's clock is a second observation the level takes
    in. The textbook trend holds, since a move over no clock has no slope;
    the line fit's slope moves by its own gain, as a least-squares slope
    does. A row with a null target or a zero weight leaves ``l`` and ``b``
    where the last observation put them and carries its clock to the next
    one. So it gives the same numbers as if it were absent. The trend is per
    clock unit, so an irregular clock extrapolates the right distance.

    .. rubric:: Parameters

    The level forgets at the spec's ``half_life`` (or ``lam``), in clock units,
    ``inf`` included: one knob under one name, as for every model.
    ``level_half_life`` is refused, with a message naming ``half_life``.

    ``trend_half_life``
        How fast the trend forgets, in clock units, above 0. Default four times
        the level's half-life; ``inf`` is the whole history's drift, not a
        trend pinned at zero: the line fit's gain on the slope, which is the
        weighted least-squares slope exactly only when the level does not
        decay either. Beside a finite level half-life the slope learns from
        errors against a level that forgets, and lands near the line fit's:
        0.2% to 1% of it off at a level half-life of 5 over 3,000 rows.
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
    ``fit_intercept`` has nothing to act on: the level is the forecast's intercept.

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
