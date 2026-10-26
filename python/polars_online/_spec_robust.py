"""The robust regressions' builders, :func:`huber` and :func:`quantile`: a
part of :mod:`polars_online._spec` in a file of its own, keeping that one
under ``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`polars_online.spec` documents them with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from polars_online._duration import Duration
from polars_online._kwargs import CommonKwargs
from polars_online._spec import _SKIPPED_WHEN_ABSENT, TargetList, _checked, _common


@_checked
def huber(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    huber_delta: float | None = None,
    huber_scale: Literal["mad", "rms"] | None = None,
    huber_scale_half_life: float | Duration | None = None,
    ridge: float | None = None,
    standardize: bool = False,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    ridge_scale: Literal["mean", "sum"] = "mean",
    emit_huber_scale: bool = False,
    huber_refit_rows: int | None = None,
    refit_budget: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Huber regression: :func:`ewridge` with the rows that miss by far
    down-weighted.

    Each row's weight is scaled by the Huber weight of its prior residual, the
    residual of the prediction made before the row is learned, so the reweighting
    stays out-of-sample. A row that misses by more than ``huber_delta`` standard
    deviations pulls the fit less the further it misses.

    .. rubric:: The fit

    With ``e`` the row's residual, ``d = huber_delta`` and ``s`` the target's
    residual scale as the row arrives:

    .. code-block:: text

        w_robust = 1              if |e| <= d * s
                 = d * s / |e|    otherwise
        the ewridge update, at weight w * w_robust

    The weights are per target, so the sums are per target here (one accumulator
    each), where :func:`ewridge` shares one across targets.

    ``s`` is the part of the estimator that makes it robust, so it is a robust
    scale by default. With ``huber_scale = "mad"`` it is the exponentially
    weighted median of ``|e|`` divided by ``Φ⁻¹(3/4) = 0.6745``, which makes it
    the standard deviation when the errors are normal. A wild row moves a median
    by its weight, whatever its size, so the outliers cannot widen the cut that
    is meant to catch them: the scale holds until half the weight is outliers.
    The median is read from a quantile sketch per target, within 0.78% of the
    exact weighted median. The sketch's buckets span at most 16 octaves around
    the median, a factor of 65,536, and a residual beyond them is folded into
    the outermost bucket on its side, its weight kept: at most about 9 KB of state per target,
    whatever the residuals, and 14 to 17 ns per target per row. While half the
    weight or more is on residuals of exactly 0 the median is 0, and the 16
    octaves reach up from the smallest residual instead. Nothing is
    folded while the residuals span less, and the median reads what an
    unbounded sketch would unless it later moves past a bucket a residual was
    folded into, at least a factor of 256 from where it stood then: as when a
    fit's first residuals, at the target's level, are more than 65,536 times
    the noise's, and the median reads too high until the later residuals
    outweigh them. The sketch keeps each residual in units of the target's
    first, so a target rescaled by any factor, grams for kilograms, rescales
    the scale and the fit to rounding; kept in the residual's own units, a
    factor other than a power of two moved the median by up to a bucket,
    1.6%, and the predictions by up to 0.4% of the target's spread. With
    ``huber_scale = "rms"`` it is the EW root mean
    square of the residuals about 0, in which every row counts at full weight:
    one wild row sets it, and a fit's first residuals, the target's whole level
    while the intercept is still near 0, widen it for as long as the decay
    keeps them.

    Measured on ``y = 1000 + 2x + 0.5ε`` over 20,000 rows with no decay, an
    outlier of +50 on a share of the rows with ``x > 0`` (so they tilt the
    slope), the median slope over five seeds:

    .. list-table::
       :header-rows: 1

       * - outliers
         - least squares
         - ``"mad"``
         - ``"rms"``
         - statsmodels ``RLM``, MAD scale, converged
       * - 1% of rows
         - 2.415
         - 2.008
         - 2.057
         - 2.008
       * - 5%
         - 4.050
         - 2.040
         - 2.675
         - 2.039
       * - 12.5%
         - 6.975
         - 2.124
         - 4.882
         - 2.124

    .. rubric:: The refit of the first rows

    One pass weighs each row by the fit before it. Over the first rows that
    fit is close to least squares, so a share of outliers pulls it, and
    ``s`` learns the pulled fit's residuals. The cut ``d * s`` is then wide,
    the outliers keep entering at nearly full weight, and without decay the
    fit stays where they pulled it. A median over a few streams hides this:
    on the stream above, 20,000 rows at 12.5% outliers, the median stream was
    within 0.002 of statsmodels' ``RLM``, while 13 of 40 streams ended more
    than 0.05 from it, the worst by 0.43. So on the row a target's count of
    learned rows reaches ``huber_refit_rows``, those rows are refit by
    iteratively reweighted least squares, as a batch fit does:

    .. code-block:: text

        v_i = w_i * (the decay factors of the rows after i)
              v_i' the same by the scale's clock (huber_scale_half_life)
        r_i = y_i - z_i'b                  against the last pass's fit
        s   = weighted median of |r_i| at v_i', over 0.6745  (root mean square under "rms")
        u_i = min(1, d * s / |r_i|)
        b   = the ridge solve of the rows at v_i * u_i
        repeated until b moves by under 1e-10 of max|b|, or 200 passes

    The target's sums are then rebuilt from the rows at their converged
    weights, and ``s`` from the converged fit's residuals, and the stream goes
    on online. Over 20,000 rows with the refit no stream of the 40 was more
    than 0.05 from ``RLM`` at 5% to 15% outliers, and 4 at 20%, where 25 were
    without it. Over 2,000 rows the tail is longer: at 20%, 10 of 40 streams
    stayed more than 0.05 away with the default, 1 of them more than 0.25
    (by 0.32), where 29 were without it; ``huber_refit_rows=200`` left 18, 5
    of them more than 0.25 (the worst by 5.8), and ``huber_refit_rows=1000``
    left 3, none more than 0.25.
    It runs after the row's own prediction, and moves its own target's fit
    alone.

    Both scales age on every row the model sees, a row with no prediction or
    with weight 0 included, so they forget across a gap as the clock says, and
    each learns a residual at the row's own weight, not at its Huber weight.

    The cut stays in the residual's scale where the insensitivity bands of
    :func:`pa` and :func:`sgd` are in the target's own. A row beyond the cut
    still teaches, at a weight that shrinks with its miss, so a cut that early
    residuals widen does not stop the fit learning. A row inside a band
    teaches nothing, so a band that early residuals widen does: a fit from
    zero coefficients has the target's whole level for its first residuals.

    A label that arrives late is weighed when it is learned. Under an
    ``embargo``, or with a target that is a window expression looking ahead,
    a row is scored where it arrives and learned once its label is known,
    the held rows in the order they arrived. Its Huber weight is read then,
    from its residual against the fit at that moment: the fit of every row
    before it and of none after it, which is the fit it would be weighed by
    without the delay. So the delay moves the predictions, which come from
    an older fit, and leaves the fit, the weights and the scale as they are
    without it: ``coef`` under an embargo of ``E`` rows is the undelayed
    run's ``coef`` of ``E`` rows earlier, to the bit. The weight is out of
    sample either way, since the row's own label is in neither fit.

    Until a target has an ``s`` above 0, no row is down-weighted. That is
    before its first residual, or while every residual so far is exactly zero
    (under ``"mad"``, while half the weight is on such residuals), and there is
    then nothing to judge an outlier against. An ``s`` taken as 1
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
    ``huber_scale``
        The scale ``s`` the cut is drawn in: ``"mad"`` (the default), the EW
        median of ``|e|`` over 0.6745, or ``"rms"``, the EW root mean square,
        the scale this model used before this parameter existed, kept to
        reproduce those numbers.
    ``huber_scale_half_life``
        The scale's own memory, a half-life in clock units (a duration on a
        temporal clock), above 0; ``inf`` forgets nothing. Default the spec's
        ``half_life`` (or ``lam``). In a series a run of large residuals is either a burst
        of outliers or a new level, and no rule on one row tells the two apart.
        A robust scale treats every row after a level shift as an outlier until
        the median itself has moved, so it holds the fit back for longer than
        least squares does. A scale memory shorter than the fit's follows a
        shift sooner, at the price of letting a longer burst of outliers widen
        the cut. Measured with a fit half-life of 500 rows and a shift of +20
        (40 noise standard deviations): the mean absolute error over the 200
        rows from 200 rows after the shift is 13.3 for least squares, 15.1
        under ``"rms"``, 19.5 under ``"mad"`` at the fit's half-life and 14.6
        under ``"mad"`` at a half-life of 100.
    ``ridge``, ``standardize``, ``solve_every``, ``max_rows_between_solves``
        As for :func:`ewridge`: the penalty (default ``1e-6``, finite and at
        least 0), correlation-form solving, and the solve schedule. ``ridge``
        is on the mean-form Gram, in the features' squared units, and
        dimensionless under ``standardize``, which solves in correlation form.
    ``feature_sets``
        Named subsets of ``features``, each a fit of its own from the same
        sums, reported as ``pred_<t>__<set>``, as :func:`ewridge`'s are. The
        full set is fitted only when it is one of them, and a single set is
        no grid: its fields take no suffix. **A row's Huber weight comes from
        the largest set's residual**, the set with the most features, the
        first of them where two tie. The sets share the target's sums, and a
        sum holds each row at one weight, so the weight is one number per
        row and target. Weighting each set by its own residual would need a
        set of sums per set. So a smaller set's fit is the weighted least
        squares fit at the weights the largest set's residuals chose, not a
        Huber fit of its own. The residual scale ``s`` is the largest set's
        too. The largest set is the fit of a ``huber`` on its features alone,
        to the bit. The default ``min_weight`` counts every feature of the
        spec, so a largest set short of them is reported later, the fit
        unchanged.
    ``ridge_scale``
        What the ridge is scaled against, as for :func:`ewridge`: ``"mean"``
        (the default), a fixed penalty on the mean-form sums whose pull is
        permanent, or ``"sum"``, RLS's prior on the decaying sum scale,
        ``(W * S + prior_scale * ridge * I) b = W * r``, which fades as the
        rows arrive and penalizes the intercept too. ``W`` is the weight the
        target's sums hold, the Huber weights applied. With
        ``huber_delta=inf`` and ``"sum"``, the fit is :func:`rls`'s with
        ``delta=ridge``, solved on the schedule rather than on every row, so
        ``huber(ridge_scale="sum")`` is the Huber form of :func:`rls` as
        ``ewridge(ridge_scale="sum")`` is its least-squares form. ``"sum"``
        is refused beside ``standardize`` and beside two or more
        ``feature_sets``.
    ``huber_refit_rows``
        The learned rows (each of positive weight, with its target) after which each
        target's first rows are refit to convergence, once (the rubric above). Default
        ``max(500, 10 * features)`` up to 100 features, and off above 100, where the
        rows a target holds would pass 0.8 MB; a count given is always taken. Over
        20,000 rows at 20% one-sided outliers, 50 rows left 14 of 40 streams more than
        0.05 from ``RLM``, 200 and 500 left 4 and 1,000 left 2; over 2,000 rows, 200
        left 18, 500 left 10 and 1,000 left 3. A group shorter than the count never
        refits, and holds its rows to the end. ``0`` turns the refit off, which gives
        the numbers of the releases before it, to the bit. A target's rows are held
        until its refit: ``(features + 4) * 8`` bytes a learned row, and 32 bytes a run
        of rows that only aged it (its value null or its weight 0), one run for such
        rows in a row at one clock step. Past 4 runs a learned row of the count, as on
        a target present on fewer than about one row in five under an irregular clock,
        the target gives its refit up and keeps its one-pass fit. The refit takes 0.06
        to 7.2 ms once, for 1 to 50 features and 200 to 1,000 rows (an Apple M4 Pro
        under other load, best of 9). Off by default, and refused when given, beside
        ``ridge_scale = "sum"``, whose prior has aged with every row before the first,
        and beside an infinite ``huber_delta``, least squares.
    ``refit_budget``
        A bound in MiB on the refit buffers of all the spec's groups together, above
        0, or ``inf`` for none. Default 256. Each new group reserves the most its
        targets' buffers can hold when it first appears, ``(features + 20) * 8`` bytes
        a target for each row of the count (the row and its 4 runs), and keeps it
        for the life of the bank, so which groups refit depends neither on the chunking
        nor on the order the groups run in. The group that would cross the bound, and
        every new group after it, gives its refit up and keeps its one-pass fit: a
        notice says so once, and ``summary()``'s ``refit_given_up`` counts the targets.
        A group's reservation is not handed back when its refit frees its buffer, so
        once the bound is reached no later group refits. At 20 features and 500 rows a
        target reserves 160 KB, and 256 MiB refits the first 1,677 targets.
    ``emit_huber_scale``
        Emit ``huber_scale_<t>``, each target's residual scale ``s`` as the
        row's cut reads it: after the row's decay and before the row's own
        residual enters it, so the field is causal by construction and can
        serve as a weight or as the scale of a robust score of the
        residuals. Null until the target has an ``s`` above 0, and on a row
        the model does not read because a feature is null or not usable.
        Under an ``embargo`` it is read where the row is scored, as
        ``pred_<t>`` is. Default ``False``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed. The reweighted sum, which an outlier lowers, is not what it
    reads, so a stream whose warm-up meets outliers reports its first prediction
    when the rows say to. It withholds the output alone: the fit learns from
    the prediction it made on a row the gate withholds -- the scale takes its
    residual, and the row its Huber weight -- so it is the same fit at any
    ``min_weight``. A target's first prediction, reported or learned from,
    waits for its own first row per coefficient of the largest feature set,
    the model's floor, since a fit of fewer rows than unknowns is the
    ridge's alone; a ``min_weight`` below that withholds nothing more.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#huber
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#huber>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not
        null; one pair per set under ``feature_sets``, ``pred_<t>__<set>``.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
    ``huber_scale_<t>``
        Under ``emit_huber_scale``: the scale the row's cut is drawn in.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`);
        per target and set under ``feature_sets``, 0 for a feature outside the set.
    ``support_coef``
        On ``coef``'s rows, each coefficient's data share, laid out like
        ``coef``: how much of the fit after the row the data, as the loss
        weighs the rows, determined rather than the ridge
        (:mod:`polars_online.spec`); null for a feature outside the set.

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
        "huber_scale": huber_scale,
        "huber_scale_half_life": huber_scale_half_life,
        "ridge": ridge,
        "standardize": standardize,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        # `{}` is written as `[]`, which the Rust side refuses by name, as
        # `ewridge`'s is.
        "feature_sets": (
            [[k, list(v)] for k, v in feature_sets.items()] if feature_sets is not None else None
        ),
        # Written only away from their defaults, which the Rust spec skips.
        "ridge_scale": None if ridge_scale == "mean" else ridge_scale,
        "emit_huber_scale": True if emit_huber_scale else None,
        "huber_refit_rows": huber_refit_rows,
        "refit_budget": refit_budget,
    }
    # Written only when given: the Rust spec skips them when absent.
    skipped = _SKIPPED_WHEN_ABSENT["huber"]
    model = {k: v for k, v in model.items() if v is not None or k not in skipped}
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

    With ``e`` the row's residual, ``psi(e) = tau - 1{e < 0}``, ``s`` the
    target's residual scale, ``w`` the row's weight and ``h`` the band's
    half-width:

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

    The scale. ``s`` is the MAD of the target's residuals: their EW median of
    ``|e|`` over ``Phi^-1(3/4) = 0.6745``, which is the standard deviation for
    normal residuals, aged on the model's clock. A wild row moves a median by
    its weight, not by its size, so the rows the band is there to bound
    cannot widen it. In their root mean square they did: at 5% of the rows
    lifted by a hundred noise deviations, all on one side, the root mean
    square stood near 11 against a noise of 0.5, the band held every clean
    row as a least-squares row and gave every wild row a pull of the band's
    width, and every one of 50 streams of 20,000 rows ended more than 0.05
    from ``statsmodels``' ``QuantReg`` on the same rows, the worst 0.098 at
    5% and 0.98 at 20%. In the MAD none did, the worst 0.006 at 5% and 0.026
    at 20%, with or without a half-life of 5,000. The median is read from
    :func:`huber`'s sketch, which keeps each residual in units of the first,
    so a rescaled target rescales the band and the fit to rounding; in the
    residual's own units a decimal factor moved the fit by up to 0.044 of
    the target's spread.

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
        at the median and a seventh at the 0.9 quantile.
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
    of the sample, is not what it reads. It withholds the output alone: the
    fit learns from the prediction it made on a row the gate withholds, so
    it is the same fit at any ``min_weight``. A target's first prediction,
    reported or learned from, waits for its own first row per coefficient,
    the model's floor, since a fit of fewer rows than unknowns is the
    ridge's alone; a ``min_weight`` below that withholds nothing more.

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
