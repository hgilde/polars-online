"""The stochastic gradient builder, :func:`sgd`: a part of
:mod:`polars_online._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`polars_online.spec` documents it with the rest.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Unpack

from polars_online._duration import Duration
from polars_online._kwargs import CommonKwargs
from polars_online._spec import _SKIPPED_WHEN_ABSENT, TargetList, _checked, _common


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
    """Stochastic gradient descent with a choice of losses: one gradient step per
    row, no solves.

    The cheap baseline, O(k) per row where the exact solvers are O(k²), and the
    only model here that takes count targets (``loss = "poisson"``). The
    coefficients can be constrained (bounded, summed to a constant, held on the
    simplex), which makes it the model for portfolio and mixing weights.

    .. rubric:: The fit

    With ``eta = z . b``, ``p = link(eta)`` and ``d = dL / d eta``:

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

    then ``g_i = d * z_i * w + l2 * b_i`` for a slope, and ``g_0 = d * w`` for
    the intercept, which is not penalised; each ``g_i`` is clamped to ``+/-
    clip_gradient`` (times ``s_y`` under the squared and Huber losses, below);
    then ``b_i -= lr_i * g_i``. With ``standardize=False``,
    ``z = [1, x]`` and ``b`` is in the caller's units; under ``standardize``,
    the default, ``z`` is the row standardized and the step, once the scaler
    has warmed up, is mapped into the caller's units (``standardize`` below).
    The Poisson link clamps ``eta`` to ``+/- 30`` before the ``exp``, so a
    Poisson prediction never exceeds ``e ** 30``, about ``1.07e13``.

    ``s`` is the scale of the target's out-of-sample residuals ``y - p`` as the
    row arrives, as for :func:`huber`: under ``huber_scale = "mad"``, the
    default, the EW median of ``|y - p|`` over 0.6745, and under ``"rms"`` the
    EW root mean square of ``y - p``. Either learns from the rows with the
    target, a weight above 0 and a prediction, each joining after its own step,
    and ages on every row.
    ``s_y`` is the target's own EW standard deviation as the row arrives: the
    spread of ``y`` around its EW mean, over the rows with the target and a
    weight above 0, whatever the fit. Until the target has an ``s`` above 0 the
    Huber row is a squared-loss row, and until it has an ``s_y`` above 0, two
    weighted rows of different values, the tube has no width. A cut and a tube
    in the target's own units would fit one scale and fail the others: a target
    in thousandths would never leave a tube of 0.1, so it would never be
    learned, and the cut would never bind on it. With ``s`` and ``s_y``, the
    squared and Huber losses fit a target scaled by ``c`` as the unscaled one,
    scaled by ``c``.

    The cut and the tube are in different units because the fit starts from
    zero coefficients, so a residual's spread can carry the target's level.
    Beyond the cut the Huber gradient is clipped, not zero, so a cut those
    residuals widen still lets every row teach, and the median scale forgets
    them once the fit has the level. The intercept takes the level from the
    first row (the rubric on the target's level below), so the root mean
    square no longer keeps it, but the outliers still inflate it. On ``y =
    1000 + 2x + 0.5ε`` with +50 on 5% of the rows, all with ``x > 0``, and
    no decay, the median slope over five seeds is 3.751 under the squared
    loss, 2.557 under ``"rms"`` and 2.013 under ``"mad"``, with or without
    the refit below. Over 40 seeds of 20,000 rows at 5% every stream stayed
    more than 0.05 from statsmodels' converged ``RLM`` under ``"rms"``, and 36
    of 40 more than 0.25 (a median gap of 0.60), with the refit or without
    it. Inside the tube
    the gradient is zero. On a target at 1,000 with a spread of 2, the first residuals put the
    residual's std near 1,000, so a tube of ``eps = 0.1`` drawn in it would be
    about 100 wide and hold every row. Without decay it would never narrow,
    and the fit would stop where it stood.

    .. rubric:: The Huber loss's refit of the first rows

    The first steps start from zero slopes, so a share of outliers captures
    the Huber fit: they pull the first steps, ``s`` learns the pulled fit's
    residuals, the cut ``delta * s`` is wide, and the outliers keep teaching
    at nearly full weight. Over 2,000 rows at 20% one-sided outliers with no
    decay, 28 of 40 streams ended more than 0.05 in slope from statsmodels'
    converged ``RLM`` in one pass, the worst by 3.6 (at 10%, 4 of 40, the
    worst by 0.069; every one of the 40 more than 0.25 off before the
    intercept took the level from the first row). So on the row a target's count
    of learned rows reaches ``huber_refit_rows``, those rows' converged Huber
    fit replaces its coefficients:

    .. code-block:: text

        v_i = w_i * (the decay factors of the rows after i), v_i' the same
              by the scale's clock (huber_scale_half_life)
        b   = weighted least squares of the rows at v_i
        r_i = y_i - z_i'b,  s = weighted median of |r_i| at v_i', over 0.6745
        b   = weighted least squares at v_i * min(1, delta * s / |r_i|)
        repeated until b moves by under 1e-10 of max|b|, or 200 passes

    with a ridge of 1e-6 on the mean-form system, and ``l2`` as the steps
    apply it: a penalty of ``l2 * s_i^2`` on each slope (``s_i`` the
    feature's scale under ``standardize``, 1 without), once a row, so the fit
    is the minimum of ``sum(v_i * rho(r_i / s)) + l2 / 2 * sum(s_i^2 * b_i^2)
    * sum(c_i)``, ``c_i`` each row counted once and decayed as its weight
    is. The intercept is not penalized. ``s`` is then
    rebuilt from the fit's residuals. The fit is set in the coordinates the
    coefficients are held in: the caller's units, or, while the scaler still
    warms up, its moments' coordinates as they stand (``b_i * s_i`` and
    ``b_0 + sum(m_i * b_i)``), so that they read out as the fit. A constraint
    projects it as it projects a step. AdaGrad's sums of squared gradients
    stay, so the step sizes are the ones the rows taught; emptied, they would
    make the next steps throw the fit off. ``inv_scaling`` reads the weight,
    which the refit does not move. Over 2,000 rows with the refit, 2 of the
    40 streams at 10% were more than 0.05 from ``RLM``, the worst by 0.075,
    and at 20% 8, none more than 0.25 (200 rows of refit: 16, 4 of them).
    Over 20,000 rows at 20% the 31 of 40 streams one pass leaves more than
    0.05 off (the worst by 1.6) became 23 and none more than 0.25. A
    constant learning rate's own wander is about 0.03 at every share.

    .. rubric:: The target's level, and a rate past its stability bound

    The coefficients start at 0, so the first residuals are the target's
    level. Left to the steps, the level moved every slope by about
    ``learning_rate`` times the level: under ``"inv_scaling"`` at a rate of 2
    with power 1, on two features and 50 seeds, the worst stream was 2.49 off
    least squares after 8,000 rows at a level of 1,000, against 0.0096 at a
    level of 0, and a level of a million, which the clip let the steps learn
    at only the rate times ``1e3`` a row, was still 980,000 off. So under an
    identity link (every loss but ``poisson`` and ``logistic``) with an
    intercept, the intercept takes the running mean's step in place of its
    gradient's while that is the larger:

    .. code-block:: text

        W = the target's learned weight, undecayed, this row's in it;  a = w / W
        squared, huber:  b_0 += -a * d        while a > learning_rate, or n <= 22
        quantile, epsilon_insensitive:  b_0 += a * (y - p)        while n <= 22

    ``d`` the loss's gradient (the residual, clipped at the Huber cut), ``n``
    Kish's count of the rows, ``W ** 2 / sum(w ** 2)``, and 22 the count the
    scaler warms up over (``standardize``). The sign-valued losses' gradient
    is not in the target's units, so they take the residual's running mean
    for the warm-up alone. On the target's first learned row that is its
    whole residual and nothing else moves; AdaGrad's sum learns the
    intercept's gradient either way. Now a level costs rounding: 0.0077 at
    either level, and the quantile loss at a level of 3 0.043 where it was
    0.21. A target with no level loses nothing measurable: 200-row groups of
    a zero-mean target, 20 features, a constant rate of 0.01, scored R² 0.46
    at rows 25-50 (0.45 before), and 0.44 at a level of 10 per group
    (-47,000 before).

    A step multiplies its own row's residual by ``1 - q``, with

    .. code-block:: text

        q = learning_rate * (w * c * |z|^2 + l2)

    ``z`` the row the step is taken on (the intercept's 1 in it) and ``c``
    the loss's curvature at the prediction: 1 for ``squared`` and inside the
    Huber cut, 0 beyond it, ``p`` for ``poisson``, ``p * (1 - p)`` for
    ``logistic``. The residual shrinks in mean square while ``E[(1 - q)^2] <
    1``, that is while ``E[q^2] < 2 * E[q]``. Each target keeps both sums
    over its stepped rows, each decayed by ``2 ** (-1/100)`` a row, and while
    the squares' sum is above twice the gains' its predictions are withheld,
    with ``withheld_reason`` ``unstable_rate`` and a notice the first time.
    The fit learns every row as before. On standardized independent normal
    features the rule is least mean squares' bound on a constant rate,
    ``2 * (k + 1) / (k ** 2 + 4 * k + 1)`` for ``k`` features with an
    intercept: 0.261 at five, where a measured fit stayed put at 0.25 and
    diverged at 0.3. Past it the fit diverged until ``clip_gradient`` held
    it, finite and wrong: at 0.5 the last 1,000 of 20,000 rows' error was
    50,000 times the target's variance, and nothing said so. ``"inv_scaling"``
    from a rate past the bound is withheld on its first rows and released as
    the rate falls, once the large gains of those rows have faded from the
    sums, whose half-life is 100 stepped rows. ``"adagrad"`` moves a
    coefficient at most the rate a row, and the quantile and
    epsilon-insensitive steps are a sign times the rate, so none of them is
    gated.

    .. rubric:: Parameters

    ``loss``
        One of the table's. Default ``"squared"``. ``epsilon_insensitive`` has
        a sign-valued subgradient, so a constant rate oscillates in a band
        around the optimum; use ``schedule = "inv_scaling"`` with it.
        ``poisson`` takes counts: a chunk with a negative target is refused,
        naming the row, before any stream is touched, as scikit-learn's
        ``PoissonRegressor`` refuses one. Taken as it stands, a negative
        target's ``p - y`` would drive the prediction down to the link's floor,
        ``e ** -30``, and hold it there. ``-0.0`` is the count 0.
    ``huber_delta``
        The Huber cut, in units of ``s``, the residuals' scale, as for
        :func:`huber`, with its default, 1.345; above 0. ``inf`` clips
        nothing, which is the squared loss.
    ``huber_scale``, ``huber_scale_half_life``
        The scale ``s`` and its own memory, as for :func:`huber`: ``"mad"``
        (the default) or ``"rms"``, and a half-life defaulting to the spec's
        ``half_life`` (or ``lam``). ``loss = "huber"`` only; refused beside another loss.
        The gradient is clipped at ``huber_delta * s``, so under ``"mad"`` a
        change of level many times the noise is followed at most
        ``learning_rate * huber_delta * s`` a row until the median has
        crossed. With a fit half-life of 500 rows and a +20 shift, the mean
        absolute error 200-400 rows after it was 17.33 under ``"mad"`` at
        the fit's memory, 2.45 under ``"rms"``, 2.99 with
        ``huber_scale_half_life = 100`` and 1.41 with 20. On a stream whose
        level can change, give it about a fifth of ``half_life``, or
        shorter.
    ``huber_refit_rows``
        The learned rows after which each target's first rows are refit to their
        converged Huber fit, once (the rubric above). Default ``max(500, 10 *
        features)`` under ``loss = "huber"`` up to 100 features, and off above 100;
        a count given is always taken. ``0`` turns the refit off, which gives the
        numbers of the releases before it, to the bit. A target's rows are held until
        its refit: ``(features + 4) * 8`` bytes a learned row, and 32 bytes a run of
        rows that only aged it (its value null or its weight 0), one run for such rows
        in a row at one clock step. Past 4 runs a learned row of the count, as on a
        target present on fewer than about one row in five under an irregular clock,
        the target gives its refit up and keeps its one-pass fit. The refit takes 0.06
        to 7.2 ms once, for 1 to 50 features and 200 to 1,000 rows (an Apple M4 Pro
        under other load, best of 9). Off by default, and refused when given, beside
        an infinite ``huber_delta``, the squared loss. ``loss = "huber"`` only;
        refused beside another loss.
    ``refit_budget``
        A bound in MiB on the refit buffers of all the spec's groups together, as for
        :func:`huber`: above 0, or ``inf`` for none; default 256. Each new group
        reserves ``(features + 20) * 8`` bytes a target for each row of the count when
        it first appears, and the group that would cross the bound, and every new group
        after it, keeps its one-pass fit, with a notice and ``summary()``'s
        ``refit_given_up``. ``loss = "huber"`` only; refused beside another loss.
    ``quantile``
        The level for ``loss = "quantile"``, in ``(0, 1)``; required for it.
    ``eps``
        The half-width of the insensitive tube, in units of ``s_y``, the
        target's own std; at least 0. Default 0.01: errors under 1% of the
        target's own spread do not move the fit. In units of the noise, on a
        target a fit predicts to R², the tube is ``eps / sqrt(1 - R²)`` noise
        standard deviations wide: at 0.01, 0.067 of them at R² 0.978 and 2.2 at
        0.99998. Inside it the gradient is zero, so a tube many noise standard
        deviations wide holds the fit wherever it first lands inside it.
        Measured (one feature, no decay; the out-of-sample error above the
        noise, in noise variances, median over seeds):

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

        At 0.1 a fit of a target predicted to within 1% of its spread stopped
        as soon as every error was inside the tube, about 0.08 off the truth
        in intercept and slope together.
    ``learning_rate``, ``schedule``, ``power``
        The rate (default 0.01) and its schedule: ``"constant"``,
        ``"inv_scaling"`` (``lr / (1 + lam * weight_sum) ** power``, ``lam``
        the row's decay, ``power`` default 0.5) or ``"adagrad"`` (``lr /
        (sqrt(G_i) + eps)``, ``eps`` ``1e-8`` times ``s_y`` under the
        squared and Huber losses and ``1e-8`` under the others, the step 0
        where the denominator is). The rate is per gradient coordinate: a step
        is ``lr * g_i``, so ``lr`` is in the coefficient's units over the
        gradient's, in standardized units under ``standardize``. That is one
        over the feature squared for the squared and Huber losses, and the
        target over the feature squared for the sign-valued quantile and
        epsilon-insensitive ones. A sign-valued step
        moves the fit's level by the rate times the row's weight, so at the
        default 0.01 a level the target moves to by 1,000 is 100,000
        unit-weight rows away; under ``"inv_scaling"`` at a rate of 0.5,
        without decay, about 1,000,000. The level it starts at is the
        running mean of its first rows (the rubric above). AdaGrad's sum of
        squared gradients and ``weight_sum`` both
        decay on the model's clock, so an annealed or adapted rate opens up
        again after a long gap instead of staying frozen. The coefficients
        themselves do not decay: every row's step moves them, so under
        ``"constant"`` their memory is in rows, about ``1 / (lr * E[z**2])`` of
        them, whatever the clock between rows. ``half_life`` reaches
        ``weight_sum`` and ``min_weight``, the scaler and AdaGrad's sum, not
        the coefficients. ``learning_rate`` is above 0, and ``power`` at least
        0.
    ``l2``
        A ridge on every step, on the slopes only: the intercept is not
        penalised. Under ``standardize`` it is on the slope in the row's
        standardized coordinates, ``s_i * b_i``. Default 0.0, and at least 0.
    ``clip_gradient``
        A cap on each coordinate of the gradient, which clamps each ``g_i``: a
        box, not a cap on the gradient's norm. Under ``squared`` and ``huber``,
        whose gradient is in the target's units, the box is ``+/-
        clip_gradient * s_y``, ``s_y`` the target's own EW standard deviation
        as the row arrives, and there is none before the target has a spread;
        under the other losses, whose gradient is a sign, a count's residual or
        a probability's, it is ``+/- clip_gradient``. Either way the cap is the
        same at every scale of the target: a constant in the target's units
        clipped every gradient of a target in the millions at the default
        (0.59 to 6.5 of its spread off the unscaled fit, scaled). Default
        ``1e3``, not off, because ``poisson`` needs it: ``p = exp(eta)``, so a
        row that pushes ``eta`` up makes the next gradient exponentially larger
        and a constant rate diverges within a few thousand rows. It does not
        bind at ordinary scales for an identity-link fit. Above 0; ``inf``
        clips nothing.
    ``standardize``
        Take the step in standardized coordinates, which is the difference
        between one learning rate for every column and one per scale. Default
        ``True``, as for :func:`kalman`: raw, features times 100 take the
        defaults' fit from an R² of 0.96 to -71847. Each row is standardized
        against the running moments with the row admitted, sklearn's
        ``StandardScaler.partial_fit`` then ``transform``:
        ``z_i = (x_i - m_i) / s_i``, ``m_i`` and ``s_i`` the EW mean and
        standard deviation, ``s_i = 1`` for a feature with no spread yet.
        Admitting the row is not a leak: the rule is about the target, and the
        features of the row being predicted are known. Against the moments from
        before the row, a variance a few rows old can be tiny by chance, and
        one step throws a coefficient the rest of a short group never brings
        back. Admitted, the row's standardized value is at most
        ``sqrt(weight_sum)``. Without an intercept the row is scaled by each
        column's root mean square and not centred, ``z_i = x_i / s_i``, as for
        :func:`ewridge`, and the bound is ``sqrt(weight_sum + 1)``.

        The moments warm up first. Until Kish's count of the rows they have
        learned, ``(sum w) ** 2 / sum w ** 2`` undecayed, reaches 22, the
        coefficients are held in the standardized coordinates. The same
        standardized row serves the prediction and the step, and the
        coefficients are read out in the caller's units through the moments as
        they stand. On the row the count reaches 22 they are read out so once,
        and from then on held in the caller's units, the step taken in ``z``
        mapped into them by the row's own ``m_i`` and ``s_i``:

        .. code-block:: text

            eta   = b_0 + sum_i b_i * x_i
            g_i   = d * z_i * w + l2 * s_i * b_i,    g_0 = d * w      (each clamped)
            b_i  -= lr_i * g_i / s_i
            b_0  -= lr_0 * g_0 - sum_i m_i * lr_i * g_i / s_i

        So a step moves the row's own prediction as the step in ``z`` does, and
        one learning rate suits every column. The prediction reads no moment,
        so the moments' moving moves none of it. A fit read through the moments
        as they stand instead has a finite half-life's own wander move every
        prediction. At a half-life of 10 and R² 0.978 its slope came out 3.03
        for a truth of 2. At a half-life of 50 and R² 0.99998 its out-of-sample
        error was 248 noise variances above the noise, where the raw fit's was
        0.010. The warm-up is there because a step taken in the caller's units
        against a scale a few rows old is kept for good: held in the caller's
        units from the first row, short groups lost everything (R² -207 at rows
        25-50 of 200-row groups).
    ``coef_min``, ``coef_max``, ``coef_sum``
        Constraints on the slopes; the intercept is always free. ``coef_min``
        and ``coef_max`` bound each slope (one number for every feature, or a
        list with one entry per feature; ``-inf`` / ``inf`` for no bound), and
        ``coef_sum`` fixes the slopes' total. After each update the slopes are
        replaced by the nearest point of the feasible set, the Euclidean
        projection: ``b_i = clamp(b_i - mu, lo_i, hi_i)``. ``mu`` is 0 for a
        box alone and, with a sum, the one value at which the sum holds, found
        exactly by sorting the ``2k`` breakpoints where a coordinate meets a
        bound. ``coef_min = 0, coef_sum = 1`` puts the slopes on the simplex:
        portfolio weights, mixing weights, an ensemble over forecasts. The
        starting point (all zero) is projected too, so a simplex starts
        uniform. With ``standardize`` the step and the projection are taken in
        standardized coordinates with the bounds and the sum carried over
        exactly, and the reported coefficients satisfy the constraint in the
        caller's units after every learned row. Past the scaler's warm-up the
        coefficients are held in the caller's units and the projection is the
        same one written in them, ``b_i = clamp(b_i - mu / s_i ** 2, lo_i,
        hi_i)``, the bounds met exactly. The intercept keeps its standardized
        value, taking ``sum_i m_i * (b_i - b_i')`` from the slopes' move. A sum
        the bounds cannot reach, a floor above a cap, or an infinite bound on
        the wrong side is refused by name; floors of ``[0.1, 0.2, 0.3]`` accept
        a sum of ``0.6`` although they add up to ``0.6000000000000001``.
    ``strict_binary``
        Under ``loss = "logistic"``, refuse a chunk whose target is not 0 or 1,
        naming the row, before any stream is touched, as :func:`ftrl` does.
        Default ``False``: such a target is clamped into ``[0, 1]``, since a
        label of 5 taken as it stands would push the linear predictor up on
        every row for ever. Refused beside another loss.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed, where ``weight_sum`` counts every row, so rows with a
    null target do not warm up coefficients they never moved. A label
    ``strict_binary`` refuses is not one of them.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#sgd
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#sgd>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature, in the caller's
        units, the constraints satisfied after every learned row
        (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. ``pred_<t>`` is a probability under ``"logistic"`` and a rate
    under ``"poisson"``, and ``emit_metrics`` reads a logistic fit as
    probabilities against labels. A Poisson fit's ``hit_rate_<t>`` is null: a rate
    is positive and a count is never negative, so there is no sign to hit, and
    the sign test about zero read 1.0 whatever the fit.

    .. rubric:: Example

    .. code-block:: python

        weights = po.spec.sgd(
            "w", targets=["y"], features=["signal_a", "signal_b", "x0"], half_life=200.0,
            loss="squared",
            learning_rate=0.01, schedule="constant",
            coef_min=0.0, coef_sum=1.0,    # the slopes on the simplex: long-only, fully invested
            coef_every=1,
        )
        fit = po.ModelBank([weights]).fit_predict(df)
        last = fit["w"].struct.field("coef").drop_nulls()[-1]
        assert min(last[1:]) >= 0.0 and abs(sum(last[1:]) - 1.0) < 1e-12

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``quantile`` is required
    for ``loss = "quantile"``, ``learning_rate`` refuses ``inf``,
    and each parameter is held to the range its entry gives. The constraints
    are checked as above: a ``coef_min`` or ``coef_max`` list has one entry
    per feature, none ``NaN``, and ``coef_sum`` is finite. Under
    ``"logistic"`` and ``"poisson"`` a window target is refused, since the
    model reads a label or a count. ``huber_delta``, ``quantile``, ``eps``
    and ``strict_binary`` belong to the losses ``"huber"``, ``"quantile"``,
    ``"epsilon_insensitive"`` and ``"logistic"``, and ``power`` to ``schedule =
    "inv_scaling"``: each is refused beside another (``ValueError``), rather than
    ignored.
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
            raise ValueError(
                f"spec {json.dumps(name)}: sgd {key} is for {what} {json.dumps(owner)}; "
                f"{what} {json.dumps(chosen)} does not use it"
            )
    return spec
