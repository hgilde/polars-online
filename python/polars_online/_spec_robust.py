"""The robust regressions' builders, :func:`huber` and :func:`quantile`: a
part of :mod:`polars_online._spec` in a file of its own, keeping that one
under ``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`polars_online.spec` documents them with the rest.
"""

from __future__ import annotations

from typing import Any, Unpack

from polars_online._duration import Duration
from polars_online._kwargs import CommonKwargs
from polars_online._spec import TargetList, _checked, _common


@_checked
def huber(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    huber_delta: float | None = None,
    ridge: float | None = None,
    standardize: bool = False,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Huber regression: :func:`ewridge` with the rows that miss by far
    down-weighted.

    Each row's weight is scaled by the Huber weight of its prior residual, the
    residual of the prediction made before the row is learned, so the reweighting
    stays out-of-sample. A row that misses by more than ``huber_delta`` standard
    deviations pulls the fit less the further it misses.

    .. rubric:: The fit

    With ``e`` the row's residual, ``d = huber_delta`` and ``s`` the EW root
    mean square of the target's residuals, about 0:

    .. code-block:: text

        w_robust = 1              if |e| <= d * s
                 = d * s / |e|    otherwise
        the ewridge update, at weight w * w_robust

    The weights are per target, so the sums are per target here (one accumulator
    each), where :func:`ewridge` shares one across targets.

    ``s`` is not itself robust. It is the plain EW root mean square of the
    residuals, in which the rows the cut down-weights count at full weight, so
    a burst of outliers widens the cut for the rows after it until the EW mean
    forgets them. ``huber_delta`` is in units of that std, not of a robust one
    such as a MAD. Its weight ages on every row the model sees, a row with no
    prediction or with weight 0 included, so it forgets across a gap as the
    clock says.

    The cut stays in the residual's std where the insensitivity bands of
    :func:`pa` and :func:`sgd` are in the target's own. A row beyond the cut
    still teaches, at a weight that shrinks with its miss, so a cut that early
    residuals widen does not stop the fit learning. A row inside a band
    teaches nothing, so a band that early residuals widen does: a fit from
    zero coefficients has the target's whole level for its first residuals.

    Until a target has an ``s`` above 0, no row is down-weighted. That is
    before its first residual, or while every residual so far is exactly zero,
    and there is then nothing to judge an outlier against. An ``s`` taken as 1
    there would be a cut in the target's own units, which would down-weight the
    first predicted rows of a target in millions and none of one in millionths.

    .. rubric:: Parameters

    ``huber_delta``
        The cut, in units of ``s``, above 0: a residual within it counts at
        full weight. Default 1.345, the cut at which Huber's estimator is 95%
        as efficient as least squares when the errors are normal (Huber 1981):
        with ``Z`` standard normal and ``ψ`` the residual clipped at ``±k``,
        the efficiency is ``(2Φ(k) − 1)² / E[ψ(Z)²]``, which is 0.95 at
        ``k = 1.345``. statsmodels' ``HuberT`` takes 1.345 and scikit-learn's
        ``HuberRegressor`` 1.35. ``inf`` cuts nothing, which is least squares.
        :func:`sgd`'s ``huber_delta`` is the same constant in the same unit.
    ``ridge``, ``standardize``, ``solve_every``, ``max_rows_between_solves``
        As for :func:`ewridge`: the penalty (default ``1e-6``, finite and at
        least 0), correlation-form solving, and the solve schedule. ``ridge``
        is on the mean-form Gram, in the features' squared units, and
        dimensionless under ``standardize``, which solves in correlation form.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed. The reweighted sum, which an outlier lowers, is not what it
    reads, so a stream whose warm-up meets outliers reports its first prediction
    when the rows say to.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#huber
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#huber>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).
    ``support_coef``
        On ``coef``'s rows, each coefficient's data share, laid out like
        ``coef``: how much of the fit after the row the data, as the loss
        weighs the rows, determined rather than the ridge
        (:mod:`polars_online.spec`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them.

    .. rubric:: Example

    .. code-block:: python

        hub = po.spec.huber(
            "hub", targets=["y"], features=["x0", "x1"],
            clock="t", gap_cap=300.0, half_life=600.0,
            huber_delta=2.0,    # a residual beyond 2 sigma is down-weighted
        )
        out = po.ModelBank([hub]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`).
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
    targets: TargetList,
    features: list[str],
    quantile: float,
    ridge: float | None = None,
    standardize: bool = False,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    quantile_eps: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Quantile regression at level ``quantile``: the fit that a share ``tau`` of the
    targets fall below.

    One Newton step per row on the check loss smoothed by a uniform kernel,
    linearised at the fit the row was scored with, so it stays out-of-sample. Use
    it for a conditional median, which outliers cannot pull, or for a tail, such
    as the 0.9 quantile of a return, where a mean says nothing.

    .. rubric:: The fit

    With ``e`` the row's residual, ``psi(e) = tau - 1{e < 0}``, ``s`` the EW
    root mean square of the target's residuals, about 0, ``w`` the row's
    weight and ``h`` the band's half-width:

    .. code-block:: text

        |e| <  h:  a least-squares row, with target y + 2h (tau - 1/2)
        |e| >= h:  no weight in the sums; w * 2h * psi(e) * z into the cross-moment,
                   over the band's weight, capped at |e| / (1 + u' A^-1 u)

    The cap reads the row's full leverage against the band's system ``A``, with
    ``u`` the row's deviation from the means: a nudge brings the row's own
    prediction at most to its target, and never past it.

    The smoothed loss has the same curvature on either side of the fit, so the
    rows' own history cancels and the fit converges to the quantile regression.
    Reweighting the rows by the check loss's IRLS weight, ``psi(e) / e``, does
    not converge to it. That weight is the secant of this step where the step
    is its tangent, and it is unbounded as ``e`` goes to zero; a row whose
    prior residual happened to be near zero then holds the fit near the fit it
    was scored by. Measured at the median of a skewed noise after 20 000 rows
    against ``statsmodels``' ``QuantReg``, the step is 0.005 away, inside
    ``QuantReg``'s own standard error of 0.007, where the weights settled 0.164
    short.

    The band. ``h`` is ``quantile_eps * s``. The rows inside the band are the
    curvature the step leans on, so a much narrower band converges more slowly
    and a much wider one smooths the quantile toward the mean. The band is
    never narrower than ``(k / n) ** 0.4`` of ``s``, for ``k`` coefficients and
    the target's effective sample ``n`` -- its present rows counted one each,
    decayed, whatever their weights -- which is the smoothed-quantile bandwidth
    rate. Under a half-life the band's share of the sample is a few rows, and
    the floor is what keeps the step fed there; a long stream leaves the floor
    behind. Coverage at ``quantile = 0.9`` on normal noise reads 0.895 at
    ``half_life = 30`` and 0.900 from 200 up.

    Warm-up and rebuilding. Under three rows per coefficient of the rows the
    target was present on, the fit is ordinary least squares: a Newton step
    needs a Hessian, and a band around a fit built from a handful of rows is
    not one. The same rule rebuilds the fit after a gap or a reset has aged the
    weight away. A band holding under one row per coefficient, in rows of the
    target's mean weight, takes least-squares rows too, until it holds rows
    again. That is what rebuilds a fit a row at the input bound leaves behind.
    Such a row sets the sums at its own scale, and every later row is outside
    the band. Only nudges arrive, each ``w * 2h * psi * z`` over the band's
    weight; they cannot move the fit until that weight has decayed to nothing,
    and each is then a step that outgrows the band. From one row up an outside
    row's step lands inside the band, and the floor keeps a settled band well
    clear of one row, so the rule does not fire in steady state. Both rules
    count rows, so a weight's scale reaches neither. Rules that compared the
    rows' weight with a row count would let rows at weight 100 leave the
    warm-up on their first row, where the fit reached 1e51.

    Until a target has an ``s`` above 0, its rows are least squares too. That
    is before its first residual, or while every residual so far is exactly
    zero, and the band, a width in units of ``s``, has nothing to be drawn in.
    An ``s`` taken as 1 there would draw the band past the warm-up in the
    target's own units, and throw a target in millionths far off.

    .. rubric:: Parameters

    ``quantile``
        The level ``tau``, in ``(0, 1)``; required. 0.5 is a median regression.
    ``quantile_eps``
        The band's half-width in units of ``s``, finite and above 0. Default
        0.2, at which the band holds about a sixth of the rows of normal noise
        at the median and a ninth at the 0.9 quantile.
    ``ridge``, ``standardize``, ``solve_every``, ``max_rows_between_solves``
        As for :func:`ewridge`: the penalty (default ``1e-6``, finite and at
        least 0), correlation-form solving, and the solve schedule. ``ridge``
        is on the mean-form Gram, in the features' squared units, and
        dimensionless under ``standardize``, which solves in correlation form.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed. The band's weight, which a half-life caps at the band's share
    of the sample, is not what it reads.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#quantile
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#quantile>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).
    ``support_coef``
        On ``coef``'s rows, each coefficient's data share, laid out like
        ``coef``: how much of the fit after the row the data, as the loss
        weighs the rows, determined rather than the ridge
        (:mod:`polars_online.spec`).

    plus the fields of the diagnostics switched on, as
    :mod:`polars_online.spec` describes them. ``pred_<t>`` is the conditional
    quantile, and a ``resid`` is signed against it, so a share ``tau`` of them
    are negative, and ``1 - tau`` positive, once the fit has settled.

    .. rubric:: Example

    .. code-block:: python

        med = po.spec.quantile(
            "med", targets=["y"], features=["x0", "x1"],
            clock="t", gap_cap=300.0, half_life=600.0,
            quantile=0.5,        # the level: a median regression
            quantile_eps=0.2,    # the band the step leans on, in units of sigma
        )
        out = po.ModelBank([med]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``quantile`` is required
    and must be strictly between 0 and 1.
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
