"""The stochastic gradient builder, :func:`sgd`: a part of
:mod:`polars_online._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`polars_online.spec` documents it with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from polars_online._duration import Duration
from polars_online._kwargs import CommonKwargs
from polars_online._spec import _SKIPPED_WHEN_ABSENT, TargetList, _checked, _common
from polars_online._spec_words import quoted, refusal


@_checked
def sgd(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    loss: Literal[
        "squared", "huber", "quantile", "epsilon_insensitive", "poisson", "logistic"
    ] = "squared",
    huber_delta: float | None = None,
    huber_scale: Literal["mad", "rms"] | None = None,
    huber_scale_half_life: float | Duration | None = None,
    huber_refit_rows: int | None = None,
    refit_budget: float | None = None,
    quantile: float | None = None,
    eps: float | None = None,
    learning_rate: float | None = None,
    schedule: Literal["constant", "inv_scaling", "adagrad"] = "constant",
    power: float | None = None,
    l2: float | None = None,
    clip_gradient: float | None = None,
    standardize: bool = True,
    coef_min: float | list[float] | None = None,
    coef_max: float | list[float] | None = None,
    coef_sum: float | None = None,
    strict_binary: bool = False,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Stochastic gradient descent under a choice of losses, with one gradient step a row.

    Use it for a cheap fit, O(k) a row for ``k`` features. It is also the
    model for count targets (``loss = "poisson"``, which no other model
    takes), and for coefficients that must be bounded, sum to a constant or
    lie on the simplex, as portfolio and mixing weights do. Use
    :func:`ewridge` for an exact fit in O(k²) a row, :func:`huber` or
    :func:`quantile` for a robust or quantile fit solved on running sums,
    and :func:`pa` for steps with no learning rate to tune.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    diagnostics, are in :mod:`polars_online.spec`. Here ``min_weight``
    counts the rows the target was present on, at their weights, decayed,
    where ``weight_sum`` counts every row, so rows with a null target do not
    warm up coefficients they never moved. It withholds the output alone:
    under ``loss = "huber"`` the residual scale learns from the prediction
    the fit made on a row the gate withholds, from the first row, so the fit
    is the same at any ``min_weight``. This model's own:

    ``loss``
        The loss the steps descend. It decides what the prediction is: a
        mean under ``"squared"``, a robust mean under ``"huber"``, a quantile
        under ``"quantile"``, a fit that ignores residuals inside a tube under
        ``"epsilon_insensitive"``, a rate of counts under ``"poisson"`` and a
        probability of a 0/1 target under ``"logistic"``. *The update*'s
        table gives each one's link and gradient. Default ``"squared"``.
        ``epsilon_insensitive`` has a sign-valued subgradient, so a constant
        rate oscillates in a band around the optimum. Use
        ``schedule = "inv_scaling"`` with it. ``poisson`` takes counts: a
        call that reads the target (a call that learns, or ``predict`` on a
        chunk holding the target column) refuses a chunk with a negative
        target, naming the row, before any model learns from it, as scikit-learn's
        ``PoissonRegressor`` refuses one. Taken as it stands, a negative
        target's ``p - y`` would drive the prediction down to the link's
        floor, ``e ** -30``, and hold it there. ``-0.0`` is the count 0.
    ``huber_delta``
        The Huber cut, in units of ``s``, the residuals' scale, as for
        :func:`huber`. Default 1.345, as there. Above 0. ``inf`` clips
        nothing, which is the squared loss.
    ``huber_scale``
        The scale ``s`` of the residuals, as for :func:`huber`: ``"mad"``
        (the default) or ``"rms"``. ``loss = "huber"`` only, and refused
        beside another loss.
    ``huber_scale_half_life``
        The scale's own memory, as for :func:`huber`: a half-life, in clock
        units, above 0, defaulting to the spec's ``half_life`` (or ``lam``).
        On a stream whose level can change, give it about a fifth of
        ``half_life``, or shorter (*The residuals' scale* in Notes).
        ``loss = "huber"`` only, and refused beside another loss.
    ``huber_refit_rows``
        The learned rows after which each target's first rows are refit to
        their converged Huber fit, once (*The refit of the first rows* in
        Notes). Default ``max(500, 10 * features)`` under ``loss = "huber"``
        up to 100 features, and off above 100. A count above 0 is taken at
        any number of features. It is at most ``2 ** 20``, since each
        target's buffer is allocated at its full size at its first learned
        row. ``0`` turns the refit off, leaving
        the one-pass fit of the steps. The rows' memory, and when a target
        gives its refit up, are in *The refit's memory* in Notes. Beside an
        infinite ``huber_delta``, the squared loss, the refit is off by
        default and a count above 0 is refused.
        ``loss = "huber"`` only, and refused beside another loss.
    ``refit_budget``
        A bound in MiB on the refit buffers of all the spec's groups
        together, as for :func:`huber`: above 0, or ``inf`` for none.
        Default 320. Each new group reserves its buffers when it first
        appears, and the group that would cross the bound, and every new
        group after it, keeps its one-pass fit (*The refit's memory* in
        Notes). ``loss = "huber"`` only, and refused beside another loss.
    ``quantile``
        The level for ``loss = "quantile"``, in ``(0, 1)``. Required for it,
        and refused beside another loss.
    ``eps``
        The half-width of the insensitive tube, in units of ``s_y``, the
        target's own standard deviation. At least 0. Default 0.01: residuals
        under 1% of the target's own spread do not move the fit. Inside the
        tube the gradient is zero, so a tube many noise standard deviations
        wide holds the fit wherever it first lands inside it (*The tube* in
        Notes). ``loss = "epsilon_insensitive"`` only, and refused beside
        another loss.
    ``learning_rate``, ``schedule``, ``power``
        The rate and its schedule. The rate is per gradient coordinate: a
        step is ``lr * g_i``, so ``lr`` is in the coefficient's units over
        the gradient's, in standardized units under ``standardize``. That
        is one over the feature squared for the squared, Huber, Poisson and
        logistic losses (whose gradient is a residual, a count's residual or
        a probability's), and the target over the
        feature squared for the sign-valued quantile and epsilon-insensitive
        ones. Under ``"adagrad"``, which divides the gradient by the root of
        its summed squares, ``lr`` is in the coefficient's units. The rate's
        default is 0.01, the same starting rate as
        scikit-learn's ``SGDRegressor`` (``eta0``), and it must be finite and
        above 0.
        The schedule is one of these:

        .. list-table::
           :header-rows: 1

           * - ``schedule``
             - the rate a row
           * - ``"constant"`` (the default)
             - ``lr``
           * - ``"inv_scaling"``
             - ``lr / (1 + lam * weight_sum) ** power``, ``lam`` the row's
               decay factor and ``power`` at least 0, by default 0.5, and
               refused beside another schedule
           * - ``"adagrad"``
             - ``lr / (sqrt(G_i) + 1e-8 * s_y)`` under the squared and
               Huber losses and ``lr / (sqrt(G_i) + 1e-8)`` under the
               others, ``G_i`` the coefficient's sum of squared gradients.
               The step is 0 where the denominator is.

        Past a
        stability bound a constant rate diverges, and the predictions are
        withheld (*A rate past its stability bound* in Notes). The
        coefficients themselves do not decay (*Decay and the rate* in
        Notes).
    ``l2``
        A ridge on every step, on the slopes only: the intercept is not
        penalized. Under ``standardize`` it is on the slope in the row's
        standardized coordinates, ``s_i * b_i``. Default 0.0, no ridge, and
        at least 0.
    ``clip_gradient``
        A cap on each coordinate of the gradient, which clamps each ``g_i``:
        a box, not a cap on the gradient's norm. Under ``squared`` and
        ``huber``, whose gradient is in the target's units, the box is
        ``+/- clip_gradient * s_y``, and there is none before the target has
        a spread. Under the other losses, whose gradient is a sign, a
        count's residual or a probability's, it is ``+/- clip_gradient``.
        Either way the cap is the same at every scale of the target
        (*The gradient's cap* in Notes). Default ``1e3``, not off, because
        ``poisson`` needs it: ``p = exp(eta)``, so a row that pushes ``eta``
        up makes the next gradient exponentially larger, and a constant rate
        diverges within a few thousand rows. It does not bind at ordinary
        scales for an identity-link fit. Above 0. ``inf`` clips nothing.
    ``standardize``
        Take the step in standardized coordinates, so that one learning rate
        suits every column. Default ``True``, as for :func:`kalman`: without
        it, features 100 times larger put the default rate past its
        stability bound (*Standardizing* in Notes). Each row is standardized
        against the running moments with the row admitted, as scikit-learn's
        ``StandardScaler.partial_fit`` then ``transform`` do:
        ``z_i = (x_i - m_i) / s_i``, ``m_i`` and ``s_i`` the exponentially
        weighted (EW) mean and standard deviation, and ``s_i = 1`` for a
        feature with no spread yet. Without an intercept the row is scaled
        by each column's root mean square and not centered,
        ``z_i = x_i / s_i``, as for :func:`ewridge`. Once the scaler has
        warmed up (*Standardizing* in Notes), a change in the moments moves
        no prediction.
    ``coef_min``, ``coef_max``, ``coef_sum``
        Constraints on the slopes. The intercept is always free. Default: no
        constraint. ``coef_min`` and ``coef_max`` bound each slope (one
        number for every feature, or a list with one entry per feature;
        ``-inf`` / ``inf`` for no bound), and ``coef_sum`` fixes the slopes'
        total. After each step the slopes are replaced by the nearest point
        of the feasible set, the Euclidean projection (*Constraints* in
        Notes). ``coef_min = 0, coef_sum = 1`` puts the slopes on the
        simplex: portfolio weights, mixing weights, an ensemble over
        predictions. The starting point (all zero) is projected too, so a
        simplex starts uniform. The reported coefficients satisfy the
        constraint in the caller's units after every learned row.
    ``strict_binary``
        Under ``loss = "logistic"``, make a call that reads the target (a
        call that learns, or ``predict`` on a chunk holding the target
        column) refuse a chunk whose target is not 0 or 1, naming the row, before any model learns
        from it, as :func:`ftrl` does. Default ``False``: such a target is
        clamped into ``[0, 1]``, since a target of 5 taken as it stands
        would push the linear prediction ``eta`` up on every row forever.
        Refused beside another loss.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#sgd
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#sgd>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, a probability under ``"logistic"`` and a
        rate under ``"poisson"``, and ``y - pred`` where the target is not
        null.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row.
    ``withheld_reason``
        Why the row's predictions were withheld, null where they were not:
        a reason of the shared gates (:mod:`polars_online.spec`), such as
        ``below_min_weight`` on the first rows under the default
        ``min_weight``, or this model's own ``unstable_rate``, where the
        learning rate is past
        its stability bound.
    ``coef``
        Per target, the intercept then one entry per feature, in the caller's
        units, after the row is learned, the constraints satisfied
        (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as
    :mod:`polars_online.spec` describes them. ``emit_metrics`` reads a
    logistic fit as probabilities against the 0/1 targets. A Poisson fit's
    ``hit_rate_<t>`` is null: a rate is positive and a count is never
    negative, so there is no sign to hit, and a sign test about zero would
    read 1.0 whatever the fit.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), and ``ValueError``
    naming the problem for:

    - a ``loss`` or ``schedule`` not one of its words;
    - a ``learning_rate`` or ``clip_gradient`` not above 0, a
      ``learning_rate`` of ``inf``, an ``l2``, ``eps`` or ``power`` below 0,
      a ``huber_delta`` or ``huber_scale_half_life`` not above 0, and a
      ``quantile`` outside ``(0, 1)``;
    - no ``quantile`` under ``loss = "quantile"``;
    - ``huber_delta``, ``huber_scale``, ``huber_scale_half_life``,
      ``huber_refit_rows`` or ``refit_budget`` beside a loss other than
      ``"huber"``, ``quantile`` beside one other than ``"quantile"``,
      ``eps`` beside one other than ``"epsilon_insensitive"``,
      ``strict_binary`` beside one other than ``"logistic"``, and ``power``
      beside a ``schedule`` other than ``"inv_scaling"``: each is refused,
      not ignored;
    - a ``huber_refit_rows`` above 0 beside an infinite ``huber_delta``, or
      past ``2 ** 20``, and a ``refit_budget`` not above 0;
    - a ``coef_min`` or ``coef_max`` list without one entry per feature, or
      with a ``NaN``; a ``coef_sum`` not finite, or one the bounds cannot
      reach; a floor above a cap; an infinite bound on the wrong side;
    - under ``"logistic"`` and ``"poisson"``, a formula target, since the
      model reads a 0/1 target or a count.

    And, from a call that learns or ``predict`` on a chunk holding the
    target column, ``ValueError`` for a chunk with a negative
    target under ``"poisson"``, or a target other than 0 or 1 under
    ``strict_binary``, naming the row.

    .. rubric:: The update

    With ``z`` the row's features, the intercept's 1 in front, ``b`` the
    coefficients, ``w`` the row's weight, ``y`` the target,
    ``eta = z . b``, ``p = link(eta)`` the prediction and
    ``d = dL / d eta``:

    .. list-table::
       :header-rows: 1

       * - loss
         - link
         - ``p``
         - ``d``
       * - ``squared``
         - identity
         - ``eta``
         - ``p - y``
       * - ``huber``
         - identity
         - ``eta``
         - ``clamp(p - y, +/- delta * s)``
       * - ``quantile``
         - identity
         - ``eta``
         - ``1{y < p} - tau``
       * - ``epsilon_insensitive``
         - identity
         - ``eta``
         - 0 within ``eps * s_y`` of ``y``, else ``sign(p - y)``
       * - ``poisson``
         - log
         - ``exp(clamp(eta, +/- 30))``
         - ``p - y``
       * - ``logistic``
         - sigmoid
         - ``sigmoid(eta)``
         - ``p - clamp(y, 0, 1)``

    then:

    .. code-block:: text

        g_i  = d * z_i * w + l2 * b_i        a slope
        g_0  = d * w                         the intercept, not penalized
        g_i  = clamp(g_i, +/- clip)          clip = clip_gradient, times s_y
                                             under the squared and Huber losses
        b_i -= lr_i * g_i

    ``delta`` is ``huber_delta``, ``tau`` the ``quantile``, and ``lr_i`` the
    rate ``schedule`` gives coefficient ``i`` on the row. ``s`` is the scale
    of the target's out-of-sample residuals ``y - p`` as the row arrives, as
    for :func:`huber`. Under ``huber_scale = "mad"``, the default, it is the
    EW median of ``|y - p|`` over 0.6745, and under ``"rms"`` the EW root
    mean square of ``y - p``. ``s_y`` is the target's own EW standard
    deviation as the row arrives. The Poisson link clamps ``eta`` to
    ``+/- 30`` before the ``exp``, so a Poisson prediction never exceeds
    ``e ** 30``, about ``1.07e13``.

    That is the step with ``standardize=False``, where ``z = [1, x]`` and
    ``b`` is in the caller's units. Under ``standardize``, the default,
    ``z`` is the row standardized. Once the scaler has warmed up, the step
    is mapped into the caller's units by the row's own ``m_i`` and ``s_i``:

    .. code-block:: text

        eta   = b_0 + sum_i b_i * x_i
        g_i   = d * z_i * w + l2 * s_i * b_i,    g_0 = d * w      (each clamped)
        b_i  -= lr_i * g_i / s_i
        b_0  -= lr_0 * g_0 - sum_i m_i * lr_i * g_i / s_i

    Under an identity link with an intercept, the intercept takes the
    running mean's step in place of its gradient's: under the squared and
    Huber losses while that is the larger or the target is in its first 22
    rows, and under the quantile and epsilon-insensitive losses over those
    rows alone (*The target's level* in Notes).

    .. rubric:: Example

    This code plants mixing weights of 0.7 and 0.3 on ``a`` and ``b``, and
    none on ``c``, fits them with the slopes held on the simplex, and prints
    each weight found:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(6)
        a, b, c = rng.standard_normal((3, 20_000))
        y = 0.7 * a + 0.3 * b + 0.1 * rng.standard_normal(20_000)  # c plays no part
        pl.DataFrame({"a": a, "b": b, "c": c, "y": y}).write_parquet("mix.parquet")

        mix = po.spec.sgd(
            "mix", targets=["y"], features=["a", "b", "c"], half_life=float("inf"),
            learning_rate=0.01,
            coef_min=0.0, coef_sum=1.0,           # the slopes on the simplex
        )
        bank = po.ModelBank([mix])
        bank.fit(pl.scan_parquet("mix.parquet"))  # learn every row, keep no output
        for r in bank.coef().filter(pl.col("term") != "intercept").iter_rows(named=True):
            print(f"{r['term']}: {r['coef']:.2f}")

    .. code-block:: text

        a: 0.70
        b: 0.30
        c: 0.00

    .. rubric:: See also

    - :func:`huber` and :func:`quantile`, the robust and quantile fits
      solved on running sums.
    - :func:`pa`, steps with no learning rate, and the same scaler and
      constraints.
    - :func:`ftrl`, a logistic or linear fit whose L1 penalty zeroes
      coefficients.

    .. rubric:: Notes

    **The residuals' scale.** ``s`` learns from the rows with the target, a
    weight above 0 and a prediction, each joining after its own step, and
    ages on every row. ``s_y`` is the spread of ``y`` around its EW mean,
    over the rows with the target and a weight above 0, whatever the fit.
    Until the target has an ``s`` above 0 the Huber row is a squared-loss
    row. Until it has an ``s_y`` above 0, which takes two weighted rows of
    different values, the tube has no width. A cut and a tube in the
    target's own units would fit one scale and fail the others: a target in
    thousandths would never leave a tube of 0.1, so it would never be
    learned, and the cut would never bind on it. With ``s`` and ``s_y``, the
    squared and Huber losses fit a target scaled by ``c`` as the unscaled
    one, scaled by ``c``.

    The cut and the tube are in different units because the fit starts from
    zero coefficients, so a residual's spread can carry the target's level.
    Beyond the cut the Huber gradient is clipped, not zero, so a cut those
    residuals widen still lets every row move the fit, and the median scale
    forgets them once the fit has the level. The intercept takes the level
    from the first row (*The target's level* in Notes), so the root mean
    square no longer keeps it, but the outliers still inflate it. Take
    ``y = 1000 + 2x + 0.5ε`` with +50 on 5% of the rows, all with ``x > 0``,
    and no decay. The median slope over five seeds is 3.751 under the
    squared loss, 2.557 under ``"rms"`` with the default refit, and 2.013
    under ``"mad"`` with or without it. Over 40 seeds of 20,000 rows at 5%,
    every stream stayed more than 0.05 from statsmodels' converged ``RLM``
    under ``"rms"``. With the default refit, 36 of the 40 stayed more than
    0.25 from it (a median gap of 0.60).

    Inside the tube the gradient is zero. On a target at 1,000 with a
    spread of 2, the first residuals put the residual's standard deviation
    near 1,000, so a tube of ``eps = 0.1`` drawn in it would be about 100
    wide and hold every row. Without decay it would never narrow, and the
    fit would stop where it stood.

    The gradient is clipped at ``huber_delta * s``, so under ``"mad"`` a
    change of level many times the noise is followed at most
    ``learning_rate * huber_delta * s`` a row until the median has crossed.
    With a fit half-life of 500 rows and a +20 shift, the mean absolute
    error 200-400 rows after it was 17.33 under ``"mad"`` at the fit's
    memory, 2.45 under ``"rms"``, 2.99 with ``huber_scale_half_life = 100``
    and 1.41 with 20.

    **The refit of the first rows.** The first steps start from zero
    slopes, so a share of outliers captures the Huber fit. They pull the
    first steps, ``s`` learns the pulled fit's residuals, the cut
    ``delta * s`` is wide, and the outliers keep moving the fit at nearly
    full weight. Over 2,000 rows at 20% one-sided outliers with no decay, 28
    of 40 streams ended more than 0.05 in slope from statsmodels' converged
    ``RLM`` in one pass, the worst by 3.6. At 10%, 4 of 40 did, the worst
    by 0.069. So on the row a target's count of learned rows reaches
    ``huber_refit_rows``, those rows' converged Huber fit replaces its
    coefficients:

    .. code-block:: text

        v_i = w_i * (the decay factors of the rows after i), v_i' the same
              by the scale's clock (huber_scale_half_life)
        b   = weighted least squares of the rows at v_i
        r_i = y_i - z_i'b,  s = weighted median of |r_i| at v_i', over 0.6745
                            (under "rms", s = sqrt(sum(v_i' * r_i^2) / sum(v_i')))
        b   = weighted least squares at v_i * min(1, delta * s / |r_i|)
        repeated until no row's fit z_i'b moves by over 1e-10 of s, or 200 passes

    The fit is the minimum of

    .. code-block:: text

        sum(v_i * rho(r_i / s)) + l2 / 2 * sum(s_i^2 * b_i^2) * sum(c_i)

    with ``c_i`` each row counted once and decayed as its weight is, so ``l2``
    acts as the steps apply it, once a row. A ridge of ``1e-6 * s_i^2`` sits on
    each slope's diagonal of the normal equations the refit solves, each sum
    divided by the weight, too, ``s_i`` the feature's scale under
    ``standardize`` and 1 without. Under ``standardize`` that is 1e-6 on the
    standardized slopes, as ``huber``'s is, so a feature's units do not move the
    fit. The intercept is penalized by neither. ``s`` is then rebuilt from the
    fit's residuals.

    The fit is set in the coordinates the coefficients are held in: the
    caller's units, or, while the scaler still warms up, its moments'
    coordinates as they stand (``b_i * s_i`` and ``b_0 + sum(m_i * b_i)``).
    So the coefficients read out as the fit. A constraint projects it as it
    projects a step. AdaGrad's sums of squared gradients stay, so the step
    sizes are the ones the rows set. Emptied, they would make the next steps
    throw the fit off. ``inv_scaling`` reads the weight, which the refit
    does not move.

    With the refit, over 2,000 rows, 2 of the 40 streams at 10% were more
    than 0.05 from ``RLM``, the worst by 0.075. At 20%, 8 of the 40 were,
    none more than 0.25. With 200 rows of refit at 20%, 16 were more than
    0.05 off and 4 more than 0.25.
    Over 20,000 rows at 20%, one pass leaves 31 of 40 streams more than 0.05
    off, the worst by 1.6, and the refit 23, none more than 0.25. A constant
    learning rate's own wander is about 0.03 at every share. The refit
    takes 0.06 to 7.2 ms once, for 1 to 50 features and 200 to 1,000 rows
    (an Apple M4 Pro under other load, best of 9).

    **The refit's memory.** A target's rows are held until its refit, at
    ``(features + 10) * 8`` bytes a learned row under the MAD, whose scale
    is read exactly from their residuals until then. Under ``"rms"`` it is
    ``(features + 4) * 8``. Rows that only aged the target (its value null
    or its weight 0) are held as runs of 32 bytes, and consecutive such
    rows at one clock step share one run. Where the runs pass 4 for each
    learned row of the count, as on a target present on fewer than about
    one row in five under an irregular clock, the target gives its refit up
    and keeps its one-pass fit.

    Under ``refit_budget``, each new group reserves ``(features + 26) * 8``
    bytes for each target and each row of the count, in each instance of
    the model. An instance is the part of the model that fits one
    half-life, so a half-life grid reserves once per value. Under ``"rms"`` it is
    ``(features + 20) * 8``. The group that would cross the bound, and every
    new group after it, keeps its one-pass fit, with a readiness warning and
    ``bank.summary()``'s ``refit_given_up``.

    **The target's level.** The coefficients start at 0, so the first
    residuals are the target's level. Left to the steps, the level moves
    every slope by about ``learning_rate`` times the level. Measured under
    ``"inv_scaling"`` at a rate of 2 with power 1, on two features and 50
    seeds, the worst stream was 2.49 off least squares after 8,000 rows at a
    level of 1,000, against 0.0096 at a level of 0. A level of a million,
    which the clip lets the steps learn at only the rate times
    ``1e3 * s_y`` a row, was still 980,000 off. So under an identity link (every loss but
    ``poisson`` and ``logistic``) with an intercept, the intercept takes the
    running mean's step in place of its gradient's, as the block gives:

    .. code-block:: text

        W = the target's learned weight, undecayed, this row's in it;  a = w / W
        squared, huber:  b_0 += -a * d        while a > the row's rate, or n <= 22
        quantile, epsilon_insensitive:  b_0 += a * (y - p)        while n <= 22

    The row's rate is ``learning_rate`` under ``"constant"`` and the rate as
    it has fallen under ``"inv_scaling"``. ``"adagrad"`` has no rate shared
    by the coefficients, so there the level's step is taken over the warm-up
    alone. ``d`` is the loss's gradient (the residual, clipped at the Huber
    cut), ``n`` Kish's effective sample size of the rows,
    ``W ** 2 / sum(w ** 2)``, and 22
    the count the scaler warms up over (*Standardizing* in Notes). The
    sign-valued losses' gradient is not in the target's units, so they take
    the residual's running mean for the warm-up alone. On the target's first
    learned row that is its whole residual, and nothing else moves.
    AdaGrad's sum learns the intercept's gradient either way. With the
    running mean's step, a level of 1,000 adds nothing to the error: 0.0077
    at either level.
    The quantile loss at a level of 3 is 0.043 off with it, against 0.21
    with the gradient's step alone. A target with no level loses nothing
    measurable: 200-row groups of a zero-mean target, 20 features and a
    constant rate of 0.01 reach R² 0.46 at rows 25-50, against 0.45 with the
    gradient's step alone. At a level of 10 per group they reach 0.44,
    against -47,000.

    **A rate past its stability bound.** A step multiplies its own row's
    residual by ``1 - q``, with

    .. code-block:: text

        q = lr * (w * c * |z|^2 + l2)

    ``lr`` is the row's rate from the schedule, and ``z`` the row the step
    is taken on (the intercept's 1 in it). ``c`` is the loss's curvature at
    the prediction: 1 for ``squared`` and inside the Huber cut, 0 beyond it,
    ``p`` for ``poisson``, ``p * (1 - p)`` for ``logistic``. The residual
    shrinks in mean square while ``E[(1 - q)^2] < 1``, that is while
    ``E[q^2] < 2 * E[q]``. Each target keeps both sums over its stepped
    rows, each decayed by ``2 ** (-1/100)`` a row. While the squares' sum is
    above twice the gains' sum, its predictions are withheld, with
    ``withheld_reason`` ``unstable_rate`` and a readiness warning the first
    time. The fit learns every row as before.

    Each gain after a target's first enters the sums bounded, at most the
    larger of 2 and four times the running mean before it. The first has no
    mean to be wide against and enters whole. A gain of 0 (a row the loss
    is flat at, with no ``l2``) says nothing of the rate and is left out. A
    gain of 2 leaves its row's residual where it was. So one wide row (a
    feature far outside its spread, a weight far above the rest) moves the
    verdict by a bounded amount, and never toward withholding while the
    mean is below 1/2. Entered whole, one row at 100 times a feature's spread held a
    stable rate's output for about 1,000 rows. A rate whose every gain is
    high still passes the bound: the mean rises fourfold a row to meet it.
    In exchange, a stream whose gains average below 1/2 and pass 2 on a few
    rows only is not withheld for them.

    The rule is the one-row mean-square contraction ``E[(1 - q)^2] < 1``. On
    standardized independent normal features with an intercept, it bounds a
    constant rate at ``2 * (k + 1) / (k ** 2 + 4 * k + 1)`` for ``k``
    features. That is 0.261 at five, where a measured fit stayed put at
    0.25 and diverged at 0.3. Past it the fit diverged until
    ``clip_gradient`` held it, finite and wrong: at 0.5 the last 1,000 of
    20,000 rows' error was 36,000 times the target's variance.

    The memory, a half-life of 100 stepped rows, is a count of rows because
    the verdict estimates an expectation from a sample of the rate's rows.
    Its error is set by the spread of the gains, which is the rows', not the
    rate's or the decay's. At this memory, at one feature, where the gains
    spread most, a constant rate at 0.9 times the bound was withheld on 0.9%
    of its rows and one at 1.2 times on 99%. At five and at twenty features,
    none at 0.9 was withheld and all at 1.1. The fit's own memory, about
    ``k / mean(q)`` rows, is a handful of rows near the bound, too few to
    estimate from. As the check's memory it withheld 6-11% of a stable
    rate's rows and served a third to a half of a diverging one's. A rate
    increased past the bound after 2,000 stable rows was withheld from 68 to
    201 rows later.

    Under ``squared`` and ``huber`` a calm stability check rests. Once a target's sums
    hold two half-lives of rows, and the gains' mean square is at most their
    mean (half the bound or less), it takes no gains. So no row forms
    ``|z|^2`` for it, and its verdict stays stable. It probes every 8th
    row's gain, and wakes where the probe passes twice the resting mean or
    where a row's residual passes 8 of the target's spreads. Awake, it takes
    every gain, and rests again after 100 of them, once calm. A rate
    increased past the bound increases its gains, so a probe finds it within
    a few rows. On five standardized features over 10 streams, a rate
    increased from half the bound to 1.15 times it was first withheld within
    a row of where an awake check withheld it. One from 0.3 to 1.5 times
    was withheld within 27 rows, 65 to 99 rows after the change, the awake
    check's range. A rate near the bound is never calm, and its check never
    rests. ``"inv_scaling"`` from a rate past the bound is withheld on its
    first rows and released as the rate falls, once the large gains of those
    rows have faded from the sums. ``"adagrad"`` moves a coefficient at most
    the rate a row, and the quantile and epsilon-insensitive steps are a
    sign times the rate, so none of them is withheld for its rate.

    **Decay and the rate.** A sign-valued step moves the fit's level by the
    rate times the row's weight. So at the default 0.01, a level the target
    moves to by 1,000 is 100,000 unit-weight rows away. Under
    ``"inv_scaling"`` at a rate of 0.5, without decay, it is about 1,000,000
    rows away. The level it starts at is the running mean of its first rows
    (*The target's level* in Notes). AdaGrad's sum of squared gradients and
    ``weight_sum`` both decay on the model's clock, so an annealed or
    adapted rate opens up again after a long gap instead of staying frozen.
    The coefficients themselves do not decay: every row's step moves them,
    so under ``"constant"`` their memory is in rows, about
    ``1 / (lr * E[z**2])`` of them, whatever the clock between rows.
    ``half_life`` reaches ``weight_sum`` and ``min_weight``, the scaler, the
    scales ``s`` and ``s_y``, and AdaGrad's sum, not the coefficients.

    **The tube.** In units of the noise, on a target a fit predicts to R²,
    the tube is ``eps / sqrt(1 - R²)`` noise standard deviations wide: at
    0.01, 0.067 of them at R² 0.978 and 2.2 at 0.99998. Measured (one
    feature, no decay; the out-of-sample error above the noise, in noise
    variances, median over seeds):

    .. list-table::
       :header-rows: 1

       * - R²
         - ``inv_scaling`` at a rate of 0.5, ``eps`` 0.01
         - at 0.1
         - ``constant`` at a rate of 0.01, ``eps`` 0.01
         - at 0.1
       * - 0.978
         - 0.015
         - 0.010
         - 0.042
         - 0.027
       * - 0.99998
         - 0.20
         - 24 (0.7 to 214)
         - 0.81
         - 17 (3.4 to 30)

    At 0.1 a fit of a target predicted to within 1% of its spread stopped as
    soon as every residual was inside the tube, about 0.08 off the truth in
    intercept and slope together.

    **The gradient's cap.** A cap that is a constant in the target's units
    clips every gradient of a target in the millions at the default, and
    leaves that fit 0.59 to 6.5 of its spread off the unscaled fit, scaled.
    The cap in units of ``s_y`` is the same at every scale of the target.

    **Standardizing.** Admitting the row to the moments is not a leak: the
    rule is about the target, and the features of the row being predicted
    are known. Against the moments from before the row, a variance a few
    rows old can be tiny by chance, and one step throws a coefficient the
    rest of a short group never brings back. Admitted, the row's
    standardized value is at most ``sqrt(weight_sum)``, and
    ``sqrt(weight_sum + 1)`` without an intercept. Without ``standardize``,
    features 100 times larger put the default rate past its stability
    bound, so every prediction is withheld as ``unstable_rate``.

    The moments warm up before the coefficients are held in the caller's
    units. Until Kish's effective sample size of the rows they have
    learned, ``(sum w) ** 2 / sum w ** 2`` undecayed, reaches 22, the
    coefficients are held in the standardized coordinates. The same
    standardized row serves the prediction and the step, and the
    coefficients are read out in the caller's units through the moments as
    they stand. On the row the count reaches 22, the coefficients are
    converted to the caller's units once, and from then on held there, the
    step taken in ``z`` mapped into them by the row's own ``m_i`` and
    ``s_i``, as *The update* shows. So a step moves the row's own prediction
    as the step in ``z`` does, and one learning rate suits every column.
    The prediction reads no moment, so a change in the moments moves none of
    it.

    The alternative, a fit read through the moments as they stand, lets a
    finite half-life's own wander in the moments move every prediction. At
    a half-life of 10 and R² 0.978 its slope came out 3.03 for a truth of 2.
    At a half-life of 50 and R² 0.99998 its out-of-sample error was 248
    noise variances above the noise, where the fit without ``standardize``
    had 0.010. The warm-up is there because a step taken in the caller's
    units against a scale a few rows old is kept for good. Held in the
    caller's units from the first row, short groups lost everything: R²
    -207 at rows 25-50 of 200-row groups.

    **Constraints.** The projection is ``b_i = clamp(b_i - mu, lo_i, hi_i)``.
    ``mu`` is 0 for a box alone and, with a sum, the one value at which the
    sum holds, found exactly by sorting the ``2k`` breakpoints where a
    coefficient meets a bound. With ``standardize`` the step and the
    projection are taken in standardized coordinates with the bounds and the
    sum carried over exactly. Past the scaler's warm-up the coefficients are
    held in the caller's units and the projection is the same one written in
    them, ``b_i = clamp(b_i - mu / s_i ** 2, lo_i, hi_i)``, the bounds met
    exactly. The intercept keeps its standardized value, taking
    ``sum_i m_i * (b_i - b_i')`` from the slopes' move. Floors of
    ``[0.1, 0.2, 0.3]`` accept a sum of ``0.6`` although they add up to
    ``0.6000000000000001``.
    """
    model: dict[str, Any] = {
        "type": "sgd",
        "loss": loss,
        "huber_delta": huber_delta,
        "huber_scale": huber_scale,
        "huber_scale_half_life": huber_scale_half_life,
        "huber_refit_rows": huber_refit_rows,
        "refit_budget": refit_budget,
        "quantile": quantile,
        "eps": eps,
        "learning_rate": learning_rate,
        "schedule": schedule,
        "power": power,
        "l2": l2,
        "clip_gradient": clip_gradient,
        "standardize": standardize,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "coef_sum": coef_sum,
        "strict_binary": strict_binary,
    }
    skipped = _SKIPPED_WHEN_ABSENT["sgd"]
    model = {k: v for k, v in model.items() if v is not None or k not in skipped}
    spec = _common(name, model, targets=targets, features=features, **common)
    # A parameter of a loss or a schedule the spec does not use is refused, as
    # a switch that is off is; each was taken and ignored (review 2026-10-05,
    # YA8). After the Rust side's checks, so an unknown loss is named first.
    problems: list[Exception] = []
    for key, value, what, owner, chosen in (
        ("huber_delta", huber_delta, "loss", "huber", loss),
        ("huber_scale", huber_scale, "loss", "huber", loss),
        ("huber_scale_half_life", huber_scale_half_life, "loss", "huber", loss),
        ("huber_refit_rows", huber_refit_rows, "loss", "huber", loss),
        ("refit_budget", refit_budget, "loss", "huber", loss),
        ("quantile", quantile, "loss", "quantile", loss),
        ("eps", eps, "loss", "epsilon_insensitive", loss),
        ("power", power, "schedule", "inv_scaling", schedule),
    ):
        if value is not None and chosen != owner:
            problems.append(
                ValueError(
                    f"spec {quoted(name)}: {key} applies only under {what}="
                    f'"{owner}", but {what} is {quoted(chosen)}, which does not use it. '
                    f'Leave {key} out, or set {what}="{owner}".'
                )
            )
    if problems:
        raise refusal(f"spec {quoted(name)}", problems)
    return spec
