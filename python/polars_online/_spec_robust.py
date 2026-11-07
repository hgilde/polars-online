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
    """Huber regression on running sums, which down-weights the rows a prediction misses by far.

    Use it in place of :func:`ewridge` where a share of the rows are
    outliers that would pull a least-squares fit. Use :func:`quantile` to fit
    a quantile of the target, and :func:`sgd` with ``loss="huber"`` where
    learning a row must take O(k) time for ``k`` features: this model takes
    O(k²).

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    diagnostics, are in :mod:`polars_online.spec`. This model's own:

    ``huber_delta``
        The cut, in units of the residuals' scale ``s``, above 0: a residual
        within it counts at full weight, and one beyond it pulls the fit less
        the further it misses. Default 1.345, the cut at which Huber's
        estimator is 95% as efficient as least squares when the errors are
        normal (*The default cut* in Notes). statsmodels' ``HuberT`` takes
        1.345 and scikit-learn's ``HuberRegressor`` 1.35. ``inf`` cuts
        nothing, which is least squares. :func:`sgd`'s ``huber_delta`` is the
        same constant in the same unit.
    ``huber_scale``
        The scale ``s`` the cut is drawn in: ``"mad"`` (the default), the
        exponentially weighted median of ``|e|`` over 0.6745, or ``"rms"``,
        the exponentially weighted root mean square of the residuals about 0.
        The MAD holds until half the weight is on outliers, where one wild row
        sets the root mean square (*The scale* in Notes).
    ``huber_scale_half_life``
        The scale's own memory, a half-life in clock units (a duration on a
        temporal clock), above 0; ``inf`` forgets nothing. Default the spec's
        ``half_life`` (or ``lam``). A scale memory shorter than the fit's
        follows a level shift sooner, at the price of letting a longer burst
        of outliers widen the cut (*A level shift* in Notes).
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
        first of them where two tie. So a smaller set's fit is the weighted
        least squares fit at the weights the largest set's residuals chose,
        not a Huber fit of its own (*Feature sets* in Notes). Refused when it
        names no set, names a set twice, holds an empty set, or names a
        column not in ``features`` or one twice in a set.
    ``ridge_scale``
        What the ridge is scaled against, as for :func:`ewridge`. ``"mean"``
        (the default) is a fixed penalty on the mean-form sums, whose pull is
        permanent. ``"sum"`` is RLS's prior on the decaying sum scale,
        ``(W * S + p * ridge * I) b = W * eta``, which fades as the rows
        arrive and penalizes the intercept too. There ``S`` and ``eta`` are
        the target's running means of ``z z'`` and ``z * y``, ``z`` being the
        features with the intercept in front. ``W`` is the weight its sums
        hold, the Huber weights applied, ``p`` the product of every decay the
        stream has applied, and ``b`` the coefficients. With
        ``huber_delta=inf`` and ``"sum"``, the fit is :func:`rls`'s with
        ``delta=ridge``, solved on the schedule rather than on every row, so
        ``huber(ridge_scale="sum")`` is the Huber form of :func:`rls` as
        ``ewridge(ridge_scale="sum")`` is its least-squares form. ``"sum"``
        is refused beside ``standardize`` and beside two or more
        ``feature_sets``.
    ``emit_huber_scale``
        Emit ``huber_scale_<t>``, each target's residual scale ``s`` as the
        row's cut reads it: after the row's decay and before the row's own
        residual enters it. So the field is out-of-sample by construction,
        and can serve as a weight or as the scale of a robust score of the
        residuals. Null until the target has an ``s`` above 0, and on a
        skipped row.
        Under an ``embargo`` it is read where the row is predicted, as
        ``pred_<t>`` is. Default ``False``.
    ``huber_refit_rows``
        The learned rows (each of positive weight, with its target) after
        which each target's first rows are refit to convergence, once, as a
        batch fit does (*The refit of the first rows* in Notes). Default
        ``max(500, 10 * features)`` up to 100 features, and off above 100,
        where a refit of ``10 * features`` rows would run for seconds a
        target. A count given is taken at any number of features. It is at
        most ``2 ** 20``, since each target's buffer is allocated at its full
        size at its first learned row. ``0`` turns the
        refit off. A group shorter
        than the count never refits, and holds its rows to the end. Off by
        default, and refused when given, beside ``ridge_scale = "sum"``,
        whose prior has aged with every row before the first, and beside an
        infinite ``huber_delta``, least squares.
    ``refit_budget``
        A bound in MiB on the refit buffers of all the spec's groups
        together, above 0, or ``inf`` for none. Default 320. Each new group
        reserves the most its targets' buffers can hold when it first
        appears, and keeps it for the life of the bank, so which groups refit
        depends neither on the chunking nor on the order the groups run in.
        The group that would cross the bound, and every new group after it,
        gives its refit up and keeps its one-pass fit: a readiness warning
        says so once, and ``summary()``'s ``refit_given_up`` counts the
        targets of each instance, the part of the model that fits one
        half-life (*The refit's memory* in Notes).

    ``min_weight`` counts the rows the target was present on, at their
    weights, decayed. The reweighted sum, which an outlier lowers, is not
    what it reads, so a stream whose warm-up meets outliers reports its first
    prediction when the rows say to. It withholds the output alone: the fit
    learns from the prediction it made on a row the gate withholds (the scale
    takes its residual, and the row its Huber weight), so it is the same fit
    at any ``min_weight``. A target's first prediction, reported or learned
    from, waits for its own first row per coefficient of the largest feature
    set, the model's floor, since a fit of fewer rows than unknowns is the
    ridge's alone. A ``min_weight`` below that withholds nothing more.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#huber
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#huber>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not
        null. Under two or more ``feature_sets``, one pair per set,
        ``pred_<t>__<set>``.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row.
    ``withheld_reason``
        Why the row's predictions were withheld, null where they were not.
    ``huber_scale_<t>``
        Under ``emit_huber_scale``: the scale the row's cut is drawn in.
    ``coef``
        Per target, the intercept then one entry per feature
        (:func:`coef_index`). Under ``feature_sets``, one per target and set,
        0 for a feature outside the set.
    ``support_coef``
        On ``coef``'s rows, each coefficient's data share, laid out like
        ``coef``: how much of the fit after the row the data, as the loss
        weighs the rows, determined rather than the ridge
        (:mod:`polars_online.spec`). Under ``feature_sets``, null for a
        feature outside the set.

    plus the fields of the diagnostics switched on, as
    :mod:`polars_online.spec` describes them.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), and ``ValueError``
    naming the problem for:

    - a ``huber_delta`` not above 0, a ``huber_scale`` other than ``"mad"``
      or ``"rms"``, and a ``huber_scale_half_life`` not above 0;
    - a ``ridge`` below 0 or not finite, a ``solve_every`` below 0, and a
      ``max_rows_between_solves`` of 0;
    - a ``feature_sets`` that names no set, names one twice, holds an empty
      set, or names a column not in ``features`` or one twice in a set;
    - a ``ridge_scale`` other than ``"mean"`` or ``"sum"``, and ``"sum"``
      beside ``standardize`` or two or more feature sets;
    - a ``huber_refit_rows`` below 0 or above ``2 ** 20``, or given beside
      ``ridge_scale = "sum"`` or an infinite ``huber_delta``;
    - a ``refit_budget`` not above 0.

    .. rubric:: The update

    Per row, ``e`` is the target's residual (the prediction made before the
    row is learned, subtracted from the target), ``d = huber_delta``, ``s``
    the target's residual scale as the row arrives, and ``w`` the row's
    weight:

    .. code-block:: text

        w_robust = 1              if |e| <= d * s
                 = d * s / |e|    otherwise
        the ewridge update, at weight w * w_robust

    The weights are per target, so the running sums are per target here,
    where :func:`ewridge` shares one set across targets. Until a target has
    an ``s`` above 0, no row is down-weighted (*The scale* in Notes).

    .. rubric:: Example

    This code fits a slope of 2 on rows of which 5% miss by +50, all where
    ``x > 0``, so they tilt a least-squares slope. It prints the slope this
    model finds beside :func:`ewridge`'s:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        x = rng.standard_normal(5_000)
        y = 1.0 + 2.0 * x + 0.5 * rng.standard_normal(5_000)
        y[(x > 0) & (rng.random(5_000) < 0.1)] += 50.0   # the outliers, all on one side
        pl.DataFrame({"x": x, "y": y}).write_parquet("outliers.parquet")

        common = dict(targets=["y"], features=["x"], half_life=float("inf"))
        bank = po.ModelBank([
            po.spec.huber("hub", **common),              # the default cut, 1.345 of the scale
            po.spec.ewridge("ls", **common),             # least squares, for comparison
        ])
        bank.fit(pl.scan_parquet("outliers.parquet"))    # learn every row, keep no output
        print("the true slope: 2.00")
        for r in bank.coef().filter(pl.col("term") == "x").iter_rows(named=True):
            print(f"{r['spec']}: slope {r['coef']:.2f}")

    .. code-block:: text

        the true slope: 2.00
        hub: slope 2.05
        ls: slope 4.10

    .. rubric:: See also

    - :func:`ewridge`, the least-squares fit on the same sums.
    - :func:`quantile`, a fit of a quantile of the target, on the same
      sums and refit.
    - :func:`sgd` with ``loss="huber"``, the same loss with a step per row
      in O(k) time.

    .. rubric:: Notes

    **Out-of-sample weights.** Each row's weight is scaled by the Huber
    weight of its residual against the prediction made before the row is
    learned, so the reweighting stays out-of-sample.

    **The default cut.** With ``Z`` standard normal and ``ψ`` the residual
    clipped at ``±k``, the efficiency of Huber's estimator against least
    squares is ``(2Φ(k) − 1)² / E[ψ(Z)²]``, which is 0.95 at ``k = 1.345``
    (Huber 1981).

    **The scale.** ``s`` is the part of the estimator that makes it robust,
    so it is a robust scale by default. With ``huber_scale = "mad"`` it is
    the exponentially weighted median of ``|e|`` divided by
    ``Φ⁻¹(3/4) = 0.6745``, which makes it the standard deviation when the
    errors are normal. A wild row moves a median by its weight, whatever its
    size, so the outliers cannot widen the cut that is meant to catch them:
    the scale holds until half the weight is outliers. The median is read
    from a quantile sketch per target, within 0.78% of the exact weighted
    median. The sketch's buckets span at most 16 octaves around the median,
    a factor of 65,536, and a residual beyond them is folded into the
    outermost bucket on its side, its weight kept. So the sketch holds at
    most about 9 KB of state per target, whatever the residuals, and takes
    14 to 17 ns per target per row. While half the weight or more is on residuals of exactly 0 the
    median is 0, and the 16 octaves reach up from the smallest residual
    instead. Nothing is folded while the residuals span less, and the median
    reads what an unbounded sketch would unless it later moves past a bucket
    a residual was folded into, at least a factor of 256 from where it stood
    then. That happens when a fit's first residuals, at the target's level,
    are more than 65,536 times the noise's, and the median reads too high
    until the later residuals outweigh them. The sketch's buckets are
    fixed at a unit of 1 until the refit of the first rows sets their unit
    at the median of its converged residuals. From then a target rescaled
    by any factor, grams for kilograms, rescales the scale and the fit to
    rounding. With the refit off, a factor other than a power of two can
    move the median by up to a bucket, 1.6%, and moved the predictions by
    up to 0.4% of the target's spread where measured.

    With ``huber_scale = "rms"`` the scale is the exponentially weighted root
    mean square of the residuals about 0, in which every row counts by its
    size. One wild row sets it, and a fit's first residuals, the target's
    whole level while the intercept is still near 0, widen it for as long as the
    decay keeps them.

    Both scales age on every row the model sees, a row with no prediction or
    with weight 0 included, so they forget across a gap as the clock says,
    and each learns a residual at the row's own weight, not at its Huber
    weight. Until a target has an ``s`` above 0, no row is down-weighted.
    That is before its first residual, or while every residual so far is
    exactly zero (under ``"mad"``, while half the weight is on such
    residuals), and there is then nothing to judge an outlier against. An
    ``s`` taken as 1 there would be a cut in the target's own units, which
    would down-weight the first predicted rows of a target in millions and
    none of one in millionths.

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

    **A level shift.** In a series a run of large residuals is either a
    burst of outliers or a new level, and no rule on one row tells the two
    apart. A robust scale treats every row after a level shift as an outlier
    until the median itself has moved, so it holds the fit back for longer
    than least squares does. Measured with a fit half-life of 500 rows and a
    shift of +20 (40 noise standard deviations), the mean absolute error over
    the 200 rows from 200 rows after the shift was:

    .. list-table::
       :header-rows: 1

       * - the fit
         - mean absolute error
       * - least squares
         - 13.3
       * - ``"rms"``
         - 15.1
       * - ``"mad"``, at the fit's half-life
         - 19.5
       * - ``"mad"``, at a ``huber_scale_half_life`` of 100
         - 14.6

    **Feature sets.** The sets share the target's sums, and a sum holds each
    row at one weight, so the weight is one number per row and target.
    Weighting each set by its own residual would need a set of sums per set.
    The residual scale ``s`` is the largest set's too. The largest set is
    the fit of a ``huber`` on its features alone, to the bit. The default
    ``min_weight`` counts every feature of the spec, so a largest set short
    of them is reported later, the fit unchanged.

    **The refit of the first rows.** One pass weighs each row by the fit
    before it. Over the first rows that fit is close to least squares, so a
    share of outliers pulls it, and ``s`` learns the pulled fit's residuals.
    The cut ``d * s`` is then wide, the outliers keep entering at nearly full
    weight, and without decay the fit stays where they pulled it. A median
    over a few streams hides this: on the stream above, 20,000 rows at 12.5%
    outliers, the median stream was within 0.002 of statsmodels' ``RLM``,
    while 13 of 40 streams ended more than 0.05 from it, the worst by 0.43.
    So on the row a target's count of learned rows reaches
    ``huber_refit_rows``, those rows are refit by iteratively reweighted
    least squares, as a batch fit does:

    .. code-block:: text

        v_i = w_i * (the decay factors of the rows after i)
              v_i' the same by the scale's clock (huber_scale_half_life)
        r_i = y_i - z_i'b                  against the last pass's fit
        s   = weighted median of |r_i| at v_i', over 0.6745  (root mean square under "rms")
        u_i = min(1, d * s / |r_i|)
        b   = the ridge solve of the rows at v_i * u_i
        repeated until no row's fit z_i'b moves by over 1e-10 of s, or 200 passes

    The target's sums are then rebuilt from the rows at their converged
    weights, and ``s`` from the converged fit's residuals, and the stream
    goes on online. It runs after the row's own prediction, and moves its
    own target's fit alone. Over 20,000 rows with the refit no stream of the
    40 was more than 0.05 from ``RLM`` at 5% to 15% outliers, and 4 at 20%,
    where 25 were without it. Over 2,000 rows the tail is longer: at 20%, 10
    of 40 streams stayed more than 0.05 away with the default, 1 of them more
    than 0.25 (by 0.32), where 29 were without it. ``huber_refit_rows=200``
    left 18, 5 of them more than 0.25 (the worst by 5.8), and
    ``huber_refit_rows=1000`` left 3, none more than 0.25. Over 20,000 rows
    at 20% one-sided outliers, 50 rows left 14 of 40 streams more than 0.05
    from ``RLM``, 200 and 500 left 4 and 1,000 left 2. Over 2,000 rows, 200
    left 18, 500 left 10 and 1,000 left 3.

    Under the MAD, while a target's refit waits, its scale is read exactly,
    as the weighted median of the residuals its buffer holds. The refit
    builds the scale's sketch from the converged residuals, its buckets in
    units of their median, so a target in other units (times 3, say) gives
    the same fit to rounding. With the refit off the sketch's buckets are
    fixed: no unit is taken from residuals as they arrive, since the first
    can all be small beside the rest, and a unit taken from them carried
    their rounding into every read. So there the scale in two units can
    differ by a bucket of the sketch, about 1.6%, and a quantile fit, whose
    band is drawn in it, by more. A power of two moves nothing. A saved
    state keeps the scale rule it was saved under, and continues to the
    bit.

    **The refit's memory.** A target's rows are held until its refit:
    ``(features + 10) * 8`` bytes a learned row under the MAD
    (``(features + 4) * 8`` under ``"rms"``). A run of rows that only aged
    it (its value null or its weight 0) takes 32 bytes, and consecutive
    such rows at the same clock step are one run. Once its runs number more
    than 4 for each row of the count, as on a target present on fewer than
    about one row in five under an
    irregular clock, the target gives its refit up and keeps its one-pass
    fit. The refit takes 0.06 to 7.2 ms once, for 1 to 50 features and 200
    to 1,000 rows (an Apple M4 Pro under other load, best of 9). Under
    ``refit_budget``, each new group reserves ``(features + 26) * 8`` bytes
    a target of each model instance for each row of the count: under the
    MAD the row, its residual, its node in the tree the MAD is read from,
    and its 4 runs, and under ``"rms"``,
    ``(features + 20) * 8``, the row and its 4 runs. A half-life grid
    reserves once per value. A group's reservation is not handed back when
    its refit frees its buffer, so once the bound is
    reached no later group refits. At 20 features and 500 rows a target
    reserves 184 KB, and 320 MiB refits the first 1,823 targets.

    **Against a band.** The cut stays in the residual's scale where the
    insensitivity bands of :func:`pa` and :func:`sgd` are in the target's
    own. A row beyond the cut still moves the fit, at a weight that shrinks
    with its miss, so a cut that early residuals widen does not stop the fit
    learning. A row inside a band moves nothing, so a band that early
    residuals widen does: a fit from zero coefficients has the target's whole
    level for its first residuals.

    **A target that arrives late.** A row's Huber weight is read when it is
    learned. Under an ``embargo``, or with a target that is a window
    expression looking ahead, a row is predicted where it arrives and learned
    once its target is known, the held rows in the order they arrived. Its
    Huber weight is read then, from its residual against the fit at that
    moment: the fit of every row before it and of none after it, which is
    the fit it would be weighed by without the delay. So the delay moves the
    predictions, which come from an older fit. It leaves the fit, the
    weights and the scale as they are without it: ``coef`` under an embargo
    of ``E`` rows is the undelayed run's ``coef`` of ``E`` rows earlier, to
    the bit. The weight is out-of-sample either way, since the row's own
    target is in neither fit.
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
    quantile_refit_rows: int | None = None,
    refit_budget: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Quantile regression on running sums, the fit that a share ``tau`` of the targets fall below.

    Use it for a conditional median, which outliers cannot pull, or for a
    tail, such as the 0.9 quantile of a return, where a mean says nothing.
    Use :func:`huber` for a mean that outliers should not pull, and
    :func:`sgd` with ``loss="quantile"`` where learning a row must take O(k)
    time for ``k`` features: this model takes O(k²).

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    diagnostics, are in :mod:`polars_online.spec`. This model's own:

    ``quantile``
        The level ``tau``, in ``(0, 1)``; required. 0.5 is a median
        regression.
    ``ridge``, ``standardize``, ``solve_every``, ``max_rows_between_solves``
        As for :func:`ewridge`: the penalty (default ``1e-6``, finite and at
        least 0), correlation-form solving, and the solve schedule. ``ridge``
        is on the mean-form Gram, in the features' squared units, and
        dimensionless under ``standardize``, which solves in correlation
        form.
    ``quantile_eps``
        The band's half-width in units of the residuals' scale ``s``, finite
        and above 0. Default 0.2, at which the band holds about a sixth of the
        rows of normal noise at the median and a seventh at the 0.9 quantile.
        A much narrower band converges more slowly, and a much wider one
        smooths the quantile toward the mean (*The band* in Notes).
    ``quantile_refit_rows``
        The learned rows (each of positive weight, with its target) after
        which each target's first rows are refit to convergence, once
        (*The refit of the first rows* in Notes). Default
        ``max(500, 10 * features)`` up to 100 features, and off above 100. A
        count given is taken up to ``2 ** 20``, since the rows are held from
        the first, but one below three rows a coefficient, the rows the
        warm-up takes as least squares, is refused. ``0`` turns the refit
        off. A group shorter than the count never refits and holds its rows
        to the end.
    ``refit_budget``
        A bound in MiB on the refit buffers of all the spec's groups
        together, above 0, or ``inf`` for none. Default 320. As for
        :func:`huber`: each new group reserves its buffers' most when it
        first appears, and keeps it for the life of the bank. The group that
        would cross the bound, and every new group after it, keeps its
        one-pass fit: a readiness warning says so once, and ``summary()``'s
        ``refit_given_up`` counts the targets of each instance (*The refit's
        memory* in Notes).

    ``min_weight`` counts the rows the target was present on, at their
    weights, decayed. The band's weight, which a half-life caps at the band's
    share of the sample, is not what it reads. It withholds the output alone:
    the fit learns from the prediction it made on a row the gate withholds,
    so it is the same fit at any ``min_weight``. A target's first
    prediction, reported or learned from, waits for its own first row per
    coefficient, the model's floor, since a fit of fewer rows than unknowns
    is the ridge's alone. A ``min_weight`` below that withholds nothing more.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#quantile
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#quantile>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, the conditional quantile, and
        ``y - pred`` where the target is not null. A residual is signed
        against the quantile, so once the fit has converged a share ``tau``
        of them are negative, and ``1 - tau`` positive.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row.
    ``withheld_reason``
        Why the row's predictions were withheld, null where they were not.
    ``coef``
        Per target, the intercept then one entry per feature
        (:func:`coef_index`).
    ``support_coef``
        On ``coef``'s rows, each coefficient's data share, laid out like
        ``coef``: how much of the fit after the row the data, as the loss
        weighs the rows, determined rather than the ridge
        (:mod:`polars_online.spec`).

    plus the fields of the diagnostics switched on, as
    :mod:`polars_online.spec` describes them.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), ``TypeError`` without
    ``quantile``, and ``ValueError`` naming the problem for:

    - a ``quantile`` not strictly between 0 and 1;
    - a ``quantile_eps`` not finite or not above 0;
    - a ``quantile_refit_rows`` other than 0 below three rows a coefficient,
      or above ``2 ** 20``, and a ``refit_budget`` not above 0;
    - a ``ridge`` below 0 or not finite, a ``solve_every`` below 0, and a
      ``max_rows_between_solves`` of 0.

    .. rubric:: The update

    One Newton step per row on the check loss smoothed by a uniform kernel,
    linearized at the fit the row was predicted with, so it stays
    out-of-sample. Per row, ``e`` is the target's residual,
    ``psi(e) = tau - 1{e < 0}``, ``s`` the target's residual scale and ``w``
    the row's weight. ``h = quantile_eps * s`` is the band's half-width
    (never below the floor of *The band* in Notes), ``A`` the band's system,
    ``u`` the row's deviation from the means, and ``z`` the features with
    the intercept in front:

    .. code-block:: text

        |e| <  h:  a least-squares row, with target y + 2h (tau - 1/2)
        |e| >= h:  no weight in the sums; w * 2h * psi(e) * z into the cross-moment,
                   over the band's weight, capped at |e| / (1 + u' A^-1 u)

    The cap reads the row's full leverage against ``A``: a nudge brings the
    row's own prediction at most to its target, and never past it. Under
    three rows per coefficient, and until a target has an ``s`` above 0,
    the fit is ordinary least squares (*Warm-up and rebuilding* in Notes).

    .. rubric:: Example

    This code fits the 0.9 quantile of a line with normal noise of spread 1,
    whose intercept is ``1 + 1.28``, and prints the line found:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(4)
        x = rng.standard_normal(20_000)
        y = 1.0 + 2.0 * x + rng.standard_normal(20_000)   # the 0.9 quantile is 2.28 + 2x
        pl.DataFrame({"x": x, "y": y}).write_parquet("tail.parquet")

        q90 = po.spec.quantile(
            "q90", targets=["y"], features=["x"], half_life=float("inf"),
            quantile=0.9,                                  # the level to fit
        )
        bank = po.ModelBank([q90])
        bank.fit(pl.scan_parquet("tail.parquet"))          # learn every row, keep no output
        print("the true line: intercept 2.28, slope 2.00")
        c = {r["term"]: r["coef"] for r in bank.coef().iter_rows(named=True)}
        print(f"found: intercept {c['intercept']:.2f}, slope {c['x']:.2f}")

    .. code-block:: text

        the true line: intercept 2.28, slope 2.00
        found: intercept 2.31, slope 1.99

    .. rubric:: See also

    - :func:`huber`, the robust fit of the mean, whose scale sketch this
      model reads.
    - :func:`ewridge`, the least-squares fit on the same sums.
    - :func:`sgd` with ``loss="quantile"``, a step per row in O(k) time.

    .. rubric:: Notes

    **Why a Newton step.** The smoothed loss has the same curvature on either
    side of the fit, so the rows' own history cancels and the fit converges
    to the quantile regression. Reweighting the rows by the check loss's
    IRLS weight, ``psi(e) / e``, does not converge to it. That weight is the
    secant of this step where the step is its tangent, and it is unbounded
    as ``e`` goes to zero. A row whose residual against the fit before it
    happened to be near zero then holds the fit near the fit it was
    predicted by. Measured at the median of a skewed noise after 20 000 rows
    against ``statsmodels``' ``QuantReg``, the step is 0.005 away, inside
    ``QuantReg``'s own standard error of 0.007, where the weights ended
    0.164 short.

    **The scale.** ``s`` is the MAD of the target's residuals: their
    exponentially weighted median of ``|e|`` over
    ``Phi^-1(3/4) = 0.6745``, which is the standard deviation for normal
    residuals, aged on the model's clock. A wild row moves a median by its
    weight, not by its size, so the rows the band is there to bound cannot
    widen it. In their root mean square they did: at 5% of the rows lifted
    by a hundred noise deviations, all on one side, the root mean square
    stood near 11 against a noise of 0.5. The band held every clean row as a
    least-squares row and gave every wild row a pull of the band's width.
    Every one of 50 streams of 20,000 rows ended more than 0.05 from
    ``statsmodels``' ``QuantReg`` on the same rows, the worst 0.098 at 5% and
    0.98 at 20%. In the MAD none did, the worst 0.006 at 5% and 0.026 at
    20%, with or without a half-life of 5,000. The median is read from
    :func:`huber`'s sketch, whose buckets are fixed at a unit of 1 until the
    refit of the first rows sets their unit at the median of its converged
    residuals. From then a rescaled target rescales the band and the fit to
    rounding. With the refit off, a decimal factor can move the scale by up
    to a bucket of the sketch, about 1.6%, and moved the fit by up to 0.044
    of the target's spread where measured.

    **The band.** ``h`` is ``quantile_eps * s``. The rows inside the band are
    the curvature the step leans on, so a much narrower band converges more
    slowly and a much wider one smooths the quantile toward the mean. The
    band is never narrower than ``(k / n) ** 0.4`` of ``s``, the
    smoothed-quantile bandwidth rate. Here ``k`` is the number of
    coefficients, and ``n`` the target's present rows counted one each,
    decayed, whatever their weights. Under a half-life the band's share of
    the sample is a few rows, and the floor is what keeps the step fed
    there. A long stream leaves the floor behind. Coverage at
    ``quantile = 0.9`` on normal noise reads 0.895 at ``half_life = 30`` and
    0.900 from 200 up.

    **Warm-up and rebuilding.** Under three rows per coefficient of the rows
    the target was present on, the fit is ordinary least squares: a Newton
    step needs a Hessian, and a band around a fit built from a handful of
    rows is not one. The same rule rebuilds the fit after a gap or a reset
    has aged the weight away. A band holding under one row per coefficient,
    in rows of the target's mean weight, takes least-squares rows too, until
    it holds rows again. That is what rebuilds a fit a row at the input
    bound leaves behind. Such a row sets the sums at its own scale, and every
    later row is outside the band. Only nudges arrive, each
    ``w * 2h * psi * z`` over the band's weight. They cannot move the fit
    until that weight has decayed to nothing, and each is then a step that
    outgrows the band. From one row up an outside row's step lands inside
    the band, and the floor keeps a band at its steady state well clear of
    one row, so the rule does not fire in steady state. Both rules count
    rows, so a weight's scale reaches neither. Rules that compared the rows'
    weight with a row count would let rows at weight 100 leave the warm-up
    on their first row, where the fit reached 1e51.

    Until a target has an ``s`` above 0, its rows are least squares too.
    That is before its first residual, or while every residual so far is
    exactly zero, and the band, a width in units of ``s``, has nothing to be
    drawn in. An ``s`` taken as 1 there would draw the band past the warm-up
    in the target's own units, and throw a target in millionths far off.

    **The refit of the first rows.** One pass takes each row as the fit
    before it sees it, and without decay a row stays in the sums as it was
    taken, at full weight. The warm-up's rows are least-squares rows, and so
    are the rows inside a band drawn around a fit that wild rows have
    pulled. So wild targets in the first rows keep pulling, where a batch
    fit reads them as outliers whose pull the band's width bounds, and they
    leave only as the rows after them outweigh them. Over 50 streams of
    8,000 rows with no decay, three targets 1e4 off at the start left the
    worst stream 0.22 from statsmodels' ``QuantReg`` on the same rows, and
    wild targets on a fifth of the first 500 rows left it 0.038 off. So on
    the row a target's count of learned rows reaches
    ``quantile_refit_rows``, those rows are refit to the minimum of the
    smoothed check loss at the band the stream would draw:

    .. code-block:: text

        v_i  = w_i * (the decay factors of the rows after i)
        r_i  = y_i - z_i'b                       against the pass's fit
        s    = weighted median of |r_i| at v_i, over 0.6745
        h    = s * max(quantile_eps, (k / n) ** 0.4)
        F(b) = sum_i v_i rho_h(r_i) + the ridge's penalty
        rho_h(r) = r**2 / (4h) + (tau - 1/2) r + h/4   if |r| < h
                 = tau * r  or  (tau - 1) * r           otherwise

    ``rho_h`` is the check loss smoothed by the uniform kernel, the loss each
    row's step is taken on. Each pass is a Newton step of ``F``: the rows
    inside the band are least-squares rows at target ``y + 2h (tau - 1/2)``,
    and each row outside adds ``w * 2h * psi(r) * z``, with no cap. A step
    that would increase ``F`` is halved until it does not, since the band's
    rows measure the curvature at the pass's fit and not at the next. A band
    of fewer rows than coefficients has no curvature to step on, and such a
    pass moves every prediction toward its row by at most the band's width
    instead. The passes stop when the step moves no row's fit ``z_i'b`` by
    more than 1e-10 of ``s``, beyond the rounding of a fit that size, or when
    no step along it lowers ``F`` past its rounding, or after 200.

    The target's sums are then rebuilt as the stream takes each row
    predicted by the converged fit at its band. Where that band holds fewer
    rows than coefficients, they are rebuilt as the majorizer's step takes
    each row instead, every row a least-squares row at its weight, so the
    sums keep a weight and solve to that fit. ``s`` is rebuilt from that
    fit's residuals, and the stream goes on online. With the refit, the same
    streams ended at most 0.0053 and 0.0085 off, beside 0.0052 from a clean
    start. A shift of every one of the first rows is no outlier among them,
    so the refit keeps them, and they leave as before: a shift of 20 in the
    first 500 rows was 0.061 off at 8,000 rows, where it was 0.093. The
    refit runs after the row's own prediction and moves its own target's fit
    alone.

    **The refit's memory.** A target's rows are held until its refit, at
    ``(features + 10) * 8`` bytes a learned row. Each row's residual is
    kept beside it for the scale: a quantile fit's band is drawn in the
    MAD, which is read exactly from the held residuals while the refit
    waits. A run of rows that only aged the target (its value null or its
    weight 0) takes 32 bytes, and consecutive such rows at the same clock
    step are one run. Once its runs number more than 4 for each row of the
    count, as on a target present on fewer than about one row in five
    under an irregular clock, the target gives its refit up and keeps its
    one-pass fit. A buffer whose rows, counted one each and decayed, are
    still fewer than the warm-up's is left as the stream built it. The refit
    takes 0.09 to 2.3 ms once, for 1 to 50 features and 500 to 1,000 rows
    (an Apple M4 Pro under other load, best of 9), and the rows after it
    cost what they did. Under ``refit_budget``, each new group reserves
    ``(features + 26) * 8`` bytes a target of each model instance (a
    half-life grid reserves once per value) for each row of the count.
    """
    model: dict[str, Any] = {
        "type": "quantile",
        "quantile": quantile,
        "ridge": ridge,
        "standardize": standardize,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        "quantile_eps": quantile_eps,
        "quantile_refit_rows": quantile_refit_rows,
        "refit_budget": refit_budget,
    }
    # Written only when given: the Rust spec skips them when absent.
    skipped = _SKIPPED_WHEN_ABSENT["quantile"]
    model = {k: v for k, v in model.items() if v is not None or k not in skipped}
    return _common(name, model, targets=targets, features=features, **common)
