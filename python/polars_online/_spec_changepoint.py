"""The changepoint detectors' builders, :func:`bocpd` and :func:`corrchange`:
a part of :mod:`polars_online._spec` moved to a file of its own, keeping
that one under ``tests/test_repo_hygiene.py``'s 250 KB cap for a source
file. :mod:`polars_online.spec` documents them with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from polars_online._duration import Duration
from polars_online._kwargs import CommonKwargs
from polars_online._spec import _checked, _common, _mirror_target


@_checked
def bocpd(
    name: str,
    *,
    features: list[str],
    hazard: float | Duration = 250.0,
    hazard_col: str | None = None,
    covariance_type: Literal["full", "diag"] = "diag",
    mean_prior: list[float] | None = None,
    mean_precision_prior: float | None = None,
    degrees_of_freedom_prior: float | None = None,
    covariance_prior: list[float] | None = None,
    robust_beta: float | None = None,
    prune_below: float | None = None,
    max_run: int | None = None,
    warm_rows: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Bayesian online changepoint detection, a posterior over how long the current
    regime has lasted.

    Use it where the age of the current regime matters: it keeps a
    distribution over the run length, the rows since the last break, so "we
    are 40 rows into a regime" comes with every row, where a test answers
    only "has something changed?". It is Adams and MacKay's (2007)
    algorithm, over the features, with no target. Use :func:`corrchange`
    for a test of a change in correlation, and a regression's
    ``emit_breaks`` for a break in its coefficients.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`polars_online.spec`. This model takes ``features`` and no
    ``targets``. ``half_life`` and ``lam`` are refused, since the run-length
    posterior is what forgets, and ``hazard`` is how fast.
    ``fit_intercept`` has nothing to act on, as there is no regression.
    ``min_weight`` withholds what is reported, never what is learned. No
    diagnostic that reads residuals applies. This model's own:

    ``hazard``
        How likely a break is, in one of two forms:

        .. list-table::
           :header-rows: 1
           :widths: 26 36 38

           * - ``hazard``
             - what it is
             - the chance of a break
           * - a number, 250 by default
             - the expected rows between breaks, on any clock
             - ``1 / hazard`` after every row
           * - a duration, such as ``"1h"``
             - ``tau``, the expected time between breaks, on a temporal
               clock
             - ``1 - exp(-dt / tau)`` for the step of ``dt`` into each row,
               before the row is read

        A number is finite and above 1. A duration is a clock parameter like
        ``half_life``. It is refused without a temporal clock, beside a clock
        parameter given as a plain number, and at 0 or below. ``dt`` is the
        step decay reads, so a gap past ``gap_cap`` counts as the cap.
        *A duration hazard* in Notes gives the rules at a step of 0.
    ``hazard_col``
        Read a per-row hazard from this column instead, which the bank reads beside
        the features as it reads a weight column. The value at a row is the expected
        rows between breaks as of that row: ``1 / value`` is the chance of a break
        after the row, between it and the next. A null, NaN, ±inf or magnitude above
        ``1e100`` falls back to ``hazard``, and any other value of 1 or less
        is refused by every call that reads the column,
        :meth:`polars_online.ModelBank.predict` included, naming the row.
        Default none. A duration ``hazard`` puts its chance before each row,
        so the two are refused together.
    ``covariance_type``
        The covariance each run's predictive keeps, in scikit-learn's words,
        both exact conjugate updates:

        .. list-table::
           :header-rows: 1
           :widths: 18 40 42

           * - ``covariance_type``
             - the predictive
             - what it adds
           * - ``"diag"`` (the default)
             - a normal-inverse-gamma per feature
             -
           * - ``"full"``
             - a normal-inverse-Wishart over all of them
             - the one that can see a break in the correlation with the
               marginals unchanged: in ``run_mode``, since ``p_change`` never
               moves, and late (*Reading p_change and run_mode* in Notes)

        While the prior is set from the first rows (the default), both are
        free of the data's units: a change of units multiplies every run's
        predictive density by one constant, which the posterior's
        normalization takes out. A prior given is in the data's units.
    ``mean_prior``, ``mean_precision_prior``, ``degrees_of_freedom_prior``, ``covariance_prior``
        The conjugate prior, under scikit-learn's ``BayesianGaussianMixture``
        names:

        .. list-table::
           :header-rows: 1
           :widths: 20 50 30

           * - parameter
             - what it is
             - default
           * - ``mean_prior``
             - ``mu_0``, one value per feature
             - the first ``warm_rows`` rows' mean
           * - ``mean_precision_prior``
             - the weight of that mean in rows, finite and above 0
             - 1.0
           * - ``degrees_of_freedom_prior``
             - the degrees of freedom, finite and above ``d + 1`` under
               ``"full"`` (where the predictive has a variance) and above 1
               under ``"diag"`` (where it has a mean)
             - the smallest integer that gives the prior's variance a mean:
               ``d + 2`` under ``"full"``, whose inverse-Wishart needs
               ``nu > d + 1``, and 3 under ``"diag"``, whose per-feature
               normal-inverse-gamma needs ``nu > 2`` whatever ``d``
           * - ``covariance_prior``
             - the prior scale of the variance as a list, in the data's units:
               one positive number ``[s]`` for ``s`` times the identity, or the
               ``d * d`` entries of a symmetric positive-definite matrix, row by
               row
             - the first ``warm_rows`` rows' covariance, its diagonal under
               ``"diag"``

        ``d`` is the number of features. ``mean_precision_prior`` and
        ``degrees_of_freedom_prior`` are in rows of the stream's mean weight.
        A prior in the data's units decides what counts as surprising, so the
        defaults are the data's own (*The prior* in Notes).
    ``robust_beta``
        Temper the ``"diag"`` predictive, so that a single outlying row is
        not read as a break: each row is weighted by ``(pi(x) / pi(mode)) **
        robust_beta`` in what the run learns and in the likelihood it passes
        on. Default none, which tempers nothing: a single 20-sigma row is
        then a break whose run carries the outlier
        in its mean. A larger power finds a real shift later, and past about
        0.2 mostly not at all. Finite and above 0, and refused beside
        ``"full"``. The tempered predictive is not free of the data's units,
        so center and scale each feature before a tempered run. In the
        query before the bank, make its z-score ``(pl.col("x") -
        po.ewm_mean("x", half_life=h)) / po.ewm_std("x", half_life=h)``
        with ``lf.online.with_windows``, which runs the window operators
        (Polars' ``with_columns`` cannot).
        *Tempering* in Notes gives the measurements.
    ``prune_below``, ``max_run``
        ``prune_below`` is the share of the mass below which a run is
        dropped (default ``1e-6``, at least 0 and below 1). ``max_run`` caps
        a run's length: the runs that reach ``max_run - 1`` rows are folded
        into the one carrying the most mass (default 10,000, at least 2). A break
        collapses the runs to a few dozen, but a stream that does not break
        spreads them over thousands of run lengths. There ``max_run`` is the
        bound on the runs kept and on a row's work (*Pruning and the capped
        run* in Notes).
    ``warm_rows``
        The learned rows held to set the prior from, where ``mean_prior`` or
        ``covariance_prior`` is left out. Default ``d + 2``, one more than
        ``d + 1``, the smallest count whose covariance is positive definite in
        general. :func:`hmm` and :func:`kmeans` name theirs the same way. At
        least 2, and refused beside both priors, which leave nothing to set.
        Where the first rows are not representative and rows are to spare,
        pass a ``warm_rows`` of tens, all from one regime (*The prior* in
        Notes).

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. Every field but ``p_change`` is read before the
    row is learned. ``p_change`` is the posterior once this row's evidence
    is in.
    `docs/OUTPUTS.md#bocpd
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#bocpd>`_
    gives each field's type and when it is null:

    ``p_change``
        ``P(r_t <= 1)`` given this row, a posterior probability: the alarm,
        spiky and as big as the break is against the prior's scale. A group's
        first row is withheld under the default ``min_weight``. With both
        priors given and ``min_weight=0`` it reads 1 however the row looks
        (*Reading p_change and run_mode* in Notes).
    ``run_mode``
        The most likely run length before the row, so ``t - run_mode`` is
        the row the current run began on. It dates a break more reliably
        than ``p_change`` flags one.
    ``run_mean``
        The posterior mean run length before the row.
    ``pred_<f>``
        The pre-row predictive mean of each feature, mixed over runs as
        ``m* + sum_r p_r (m_r - m*)``, ``m*`` the most probable run's mean,
        so a feature at a level keeps the precision of its spread.
    ``loglik``
        The row's log predictive density under that mixture.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        Always null, since nothing decays.
    ``withheld_reason``
        Why the row's fields were withheld, null where they were not:
        ``below_min_weight`` on a group's first row under the default
        ``min_weight``, which takes precedence, then ``below_warm_rows`` on
        the rest of the rows held to set the prior from.

    A row whose predictive cannot be evaluated reports nulls and is counted
    in :meth:`polars_online.ModelBank.solve_failures`. It leaves the
    posterior where it stands, except for a duration ``hazard``'s chance of
    a break over its step, which time passing applies.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), ``TypeError`` for
    ``targets``, which this model does not take, and ``ValueError`` naming
    the problem for:

    - an empty ``features``, any ``half_life`` or ``lam``, and a
      ``covariance_type`` other than ``"full"`` or ``"diag"``;
    - a number ``hazard`` of 1 or less or not finite, and a duration
      ``hazard`` with no clock, beside a plain-number clock parameter, at 0
      or below, or beside ``hazard_col``;
    - a ``mean_prior`` of the wrong length or holding a value that is not
      finite, a ``covariance_prior`` holding a value that is not finite, a
      ``mean_precision_prior`` not
      finite or not above 0, a ``degrees_of_freedom_prior`` not finite or
      at or below its floor, and a ``covariance_prior`` that is neither
      ``[s]`` nor ``d * d`` entries, or is not positive, symmetric and
      positive definite (checked whole, even under ``"diag"``, which reads
      its diagonal);
    - a ``robust_beta`` beside ``covariance_type="full"``, and one at 0,
      below it or not finite;
    - a ``prune_below`` outside ``[0, 1)``, a ``max_run`` below 2, a
      ``warm_rows`` below 2 or beside both priors, and held rows past
      256 MiB;
    - a switch of a diagnostic that reads residuals;

    and, from a call that learns or predicts, ``ValueError`` for a
    ``hazard_col`` value of 1 or less, naming the row.

    .. rubric:: The update

    Adams and MacKay's Algorithm 1, in log space. ``r`` is a run length,
    ``H = 1 / hazard`` the chance of a break after the row, and ``pi_r`` run
    ``r``'s posterior predictive for this row:

    .. code-block:: text

        growth:      P(r_t = r+1, x_1:t) = P(r_t-1 = r, x_1:t-1) pi_r (1 - H)
        changepoint: P(r_t = 0,   x_1:t) = sum_r P(r_t-1 = r, x_1:t-1) pi_r H

    Under a duration ``hazard``, the chance ``h = 1 - exp(-dt / tau)`` that a
    break fell inside the step belongs before the row. So ``h`` of every
    run's mass moves to the empty run before the row is read, and the
    recursion above runs with ``H = 0``, since no time has passed since the
    row:

    .. code-block:: text

        before the row:  P(r = 0) += h * sum_{r >= 1} P(r),   P(r) *= 1 - h for r >= 1

    Run ``r`` keeps the conjugate statistics of exactly the ``r`` rows that
    hypothesis says preceded this one in the run. So run 0 holds none, and
    its predictive is the prior's, which is what makes "a new run starts
    here" something the data can vote on. A row of weight ``w`` is learned
    at ``w / w_bar``, ``w_bar`` the mean weight of the rows learned, this one
    included, in its run's statistics and in the likelihood it passes the
    recursion. So a heavier row is more evidence of both kinds, and a
    constant multiple of every weight changes nothing. What the row reports
    is read as a row of the mean weight, which is all ``predict`` can know.

    .. rubric:: Example

    This code runs the model over a level that steps up by four standard
    deviations at row 600, and prints the row the current run began on,
    before and after the step:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(7)
        x = rng.standard_normal(1_000)
        x[600:] += 4.0                                # the level steps up at row 600
        pl.DataFrame({"x": x}).write_parquet("step.parquet")

        b = po.spec.bocpd("regime", features=["x"], hazard=250.0)
        out = pl.scan_parquet("step.parquet").online.fit_predict([b]).collect()
        out = out.unnest("regime")
        for row in [599, 650, 999]:
            began = row - out["run_mode"][row]        # the row the current run began on
            print(f"row {row}: the run began on row {began}")

    .. code-block:: text

        row 599: the run began on row 0
        row 650: the run began on row 600
        row 999: the run began on row 600

    .. rubric:: See also

    - :func:`corrchange`, tests for a change in the correlation of two
      features.
    - :func:`hmm`, a fixed number of regimes the stream moves between.
    - `docs/REGIMES.md
      <https://github.com/hgilde/polars-online/blob/main/docs/REGIMES.md>`_,
      the models of regimes and changes, side by side.

    .. rubric:: Notes

    **Reading p_change and run_mode.** Under a number ``hazard``,
    ``p_change`` is ``P(r <= 1)`` and not ``P(r = 0)``: the changepoint
    branch and the growth branch share the same predictive, which makes the
    normalized mass at ``r = 0`` exactly ``H`` on every row whatever the
    data. Under a duration ``hazard``, the chance of a break is applied
    before the row, so no mass sits at ``r = 0`` after it, and ``p_change``
    is the chance that this row began a run. There, a row whose clock step
    is 0 (a stamp shared with the last row the model did not skip) reports
    0, apart from a run's first row: a group's first row, and its first
    after a reset. With ``mean_prior`` and ``covariance_prior`` both
    given, the default ``min_weight`` withholds such a row, and with
    ``min_weight=0`` it reports a ``p_change`` of 1 there. Unless both are
    given, the first ``warm_rows`` rows report null.

    ``run_mode`` dates a break more reliably than ``p_change`` flags one. A
    break the posterior accepts a row or two late shows as a fall in
    ``run_mode``, not in ``p_change``. Over six seeds of a 5-sigma step at row 500, at the defaults,
    ``run_mode`` fell to 1 or 2 within two rows of the break on every seed,
    while the largest ``p_change`` near it ranged from 0.021 to 0.749.

    .. list-table::
       :header-rows: 1
       :widths: 44 30 26

       * - the break
         - ``p_change``
         - the run length
       * - the spread steps tenfold, under a prior of the rows' scale
         - 0.83 on the row of the break
         - found within a few rows, dated to the right row
       * - a four-sigma mean shift under a diffuse prior
         - barely lifts
         - the same
       * - a change in correlation alone
         - never moves
         - never found under ``"diag"``; under ``"full"``, found a median
           of 121 rows after the break

    **A duration hazard.** A step of 0 has no chance of a break: a row that
    shares the previous row's stamp cannot begin a run. A row of weight 0
    applies its step's chance, since time passed, and learns nothing, so
    two steps with nothing learned between them read as one step of their
    sum. Under ``prune_below = 0`` each step of 0 leaves one run of no mass
    among the runs kept, until ``max_run`` folds it.

    **Tempering.** Untempered, a single 20-sigma row reads ``p_change``
    0.91, and its run carries the outlier in its mean. Tempered, an
    outlying row is atypical under every run,
    every tempered likelihood is about 1, and nothing moves. A whole new
    regime is a run of individually forgiven rows, which is why a larger
    power finds a real shift later. Over 50 streams with a four-sigma shift
    at ``hazard=250``, a shift counts as found once ``run_mode`` dates it to
    within 5 rows, inside 150 rows:

    .. list-table::
       :header-rows: 1
       :widths: 20 80

       * - ``robust_beta``
         - the shifts found
       * - 0.1
         - all 50, the worst 6 rows late, while a 20-sigma outlier left
           ``p_change`` at its floor
       * - 0.15
         - all 50, the worst 16 rows late
       * - 0.2
         - 38, the worst 120 rows late
       * - 0.3
         - 1
       * - 0.5
         - none

    The tempered predictive's
    message ``pi ** w`` carries the constant a change of units multiplies
    the density by as ``c ** w``, and ``w`` differs from run to run, so the
    runs' odds move with the units. At a level of ``1e4`` with a spread of
    ``1e-2``, ``p_change`` moved by up to 0.86 against the same stream at
    level 0 and spread 1. ``robust_beta`` is a power on a ratio of two
    densities, and has no units.

    **The prior.** Too wide a prior, and the model goes quiet, because no
    row is surprising under a predictive that wide. Too narrow or
    off-center, and every run that begins on a row has no density for it.
    With ``mean_prior`` or ``covariance_prior`` left out, the first
    ``warm_rows`` learned rows report null and are held. Their weighted mean
    and covariance (``numpy.cov``'s with ``aweights``, the sample covariance
    at equal weights) set what was left out, and then they are read, in
    order, as rows of a model that had that prior from the start. At the
    default ``degrees_of_freedom_prior`` the prior's mean variance is then
    that covariance. A feature that does not move over those rows has no
    variance to give. It takes ``2**-52 * max(mean**2, 1)``, the spread of a
    value known to ``2**-26`` of its level, since a predictive of no spread
    has no density. Under ``"full"`` a covariance that is not positive
    definite, two features that are one, gives its diagonal. With both
    given there is no warm-up. ``degrees_of_freedom_prior`` and
    ``covariance_prior`` are ``2a`` and ``2b`` in the gamma
    parametrization, which is how Adams and MacKay give their own finance
    example (``a = 1``, ``b = 1e-4``, ``hazard = 250``: in this model's
    terms ``degrees_of_freedom_prior=2.0`` and ``covariance_prior=[2e-4]``).

    The prior is set once from the ``warm_rows`` rows and kept for the
    stream's life, so at ``d + 2`` rows its scale rests on ``d + 1``
    degrees of freedom. For one feature, a 3-row variance lies between 0.05
    and 3 times the truth nine times in ten. A warm-up that straddles a
    regime change sets a wide prior: a variance of 37 across a 10-sigma
    step, against 0.53 from rows inside one regime.

    **Pruning and the capped run.** A row takes ``O(runs * d)`` under
    ``covariance_type="diag"``, tempered or not, and ``O(runs * d³)`` under
    ``"full"``, which factors each run's scale. The run vector would grow by
    one every row, so runs below ``prune_below`` of the mass are dropped.
    The runs that reach ``max_run - 1`` rows are folded into the one of them
    carrying the most mass, the oldest among equals, which takes their mass
    and the tail's place. Its statistics are scaled back to ``max_run - 1``
    rows' weight, as a decay would scale them: the weight and scatter
    shrink, the mean stays. No run kept is longer than ``max_run - 1``
    rows, so ``run_mode`` saturates one below it, and the runs kept, at most
    ``max_run`` lengths, bound the work a row takes. The capped run is an
    exponentially weighted run, scaled by ``(max_run - 1) / max_run`` each
    row. Its mean steps by ``1 / max_run`` of the row's deviation. Its
    half-life is ``ln 2 / ln(max_run / (max_run - 1))``, about
    ``0.69 * (max_run - 1)`` rows, 6,900 at the default. The run that
    dominates a quiet stream keeps its memory at the cap however many
    younger runs ``prune_below`` drops. After a break, the post-break run
    takes its place once it reaches the cap.
    """
    model: dict[str, Any] = {
        "type": "bocpd",
        "hazard": hazard,
        "hazard_col": hazard_col,
        "covariance_type": covariance_type,
        "mean_prior": mean_prior,
        "mean_precision_prior": mean_precision_prior,
        "degrees_of_freedom_prior": degrees_of_freedom_prior,
        "covariance_prior": covariance_prior,
        "robust_beta": robust_beta,
        "prune_below": prune_below,
        "max_run": max_run,
        "warm_rows": warm_rows,
    }
    mirror = _mirror_target(name, "bocpd", features, common, "its change points are in")
    targets = [hazard_col] if hazard_col is not None else mirror
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def corrchange(
    name: str,
    *,
    features: list[str],
    kind: Literal["monitor", "sequential", "window"] = "monitor",
    window_rows: int | None = None,
    alpha: float = 0.05,
    alpha_adjust: Literal["bonferroni", "none"] = "bonferroni",
    bandwidth: int | None = None,
    scalar: bool = False,
    crit: float | None = None,
    n_perm: int | None = None,
    permute_every_rows: int | None = None,
    perm_block: int | None = None,
    norm: Literal["l1", "linf"] = "l1",
    seed: int | None = None,
    change_action: Literal["flag", "reset"] = "flag",
    monitor_rows: int | None = None,
    boundary_gamma: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Tests of whether the correlations between the feature columns have changed, by
    three kinds of test.

    Use ``kind="monitor"`` to ask whether the correlations were constant over
    each span of rows, ``kind="sequential"`` to test each row as it arrives
    against a history taken as stable, and ``kind="window"`` to ask how big
    the change between two adjacent windows is. Use :func:`bocpd` for the
    age of the current regime of the features' levels and spreads, and
    :func:`deco` to follow an equicorrelation that moves.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`polars_online.spec`. This model takes ``features``, at least two,
    and no ``targets``. Nothing decays but the optional standardizer of
    ``scalar`` and ``weight_sum`` beside it, so ``half_life`` and ``lam``
    are refused except under ``scalar``. A row's positive weight reaches only
    ``weight_sum`` and that standardizer: the span, the windows and the
    history take each row unweighted. ``fit_intercept`` has nothing to act
    on, as there is no regression, and no diagnostic that reads residuals
    applies. A parameter that belongs to another kind is refused, naming the
    kinds it applies to. This model's own:

    ``kind``
        ``"monitor"`` (the default), ``"sequential"`` or ``"window"``.
    ``window_rows``
        Required by every kind: the span ``"monitor"`` tests (at least 8),
        the history ``"sequential"`` monitors against (at least 8), or each
        of ``"window"``'s two windows (at least 3).
    ``alpha``, ``alpha_adjust``
        The level, in ``(0, 1)`` (default 0.05), and, under ``"monitor"`` and
        ``"sequential"``, how it is spread over the pairs: ``"bonferroni"``,
        the default, ``alpha / npairs``, or ``"none"``, ``alpha`` on each pair.
        Under ``"window"`` one permutation statistic covers every pair at once,
        so there is nothing to spread, and an ``alpha_adjust`` other than the
        default is refused. Under ``"sequential"`` without ``crit`` each pair's
        share is refused below ``5e-11``: the critical value is solved from the
        boundary's law, and in a smaller tail the solve's quantile is off by
        more than the 0.03 Wied and Galeano's own table is good to.
    ``bandwidth``
        ``"monitor"`` and ``"sequential"``: overrides the Bartlett bandwidth,
        ``floor(ln window_rows)``, for the span and for the history alike. Lag
        ``l`` is weighted ``1 - l / bandwidth`` for ``l < bandwidth``, so at 1
        only lag 0 is left, and 0 is refused. At most ``window_rows``: a span
        of ``window_rows`` rows has no lag past ``window_rows - 1``, so a
        larger bandwidth adds no lag. It only flattens the weights toward 1,
        where the centered lags sum to 0, and the long-run variance falls
        toward 0 with them.
    ``scalar``
        ``"monitor"`` and ``"sequential"``: in place of every pair, test the
        mean of ``u``, the equicorrelation of the standardized row
        (:func:`deco`'s ``u``), with the Bartlett long-run standard deviation
        of ``u`` in place of ``D``: one statistic however many columns there
        are.
        ``half_life``/``lam`` parametrize that standardizer, and
        ``weight_sum`` with it, and are accepted only here. Left out, nothing
        decays. Default ``False``.
    ``crit``
        ``"window"``: a fixed threshold, in place of the permutation quantile.
        ``"sequential"``: replaces Wied & Galeano's critical value. Finite
        and above 0. Default none.
    ``n_perm``, ``permute_every_rows``, ``perm_block``
        ``"window"``'s permutation quantile: ``n_perm`` draws (default 200, at most
        2^20, and without ``crit`` at least 20 and at least ``ceil(1 / alpha) -
        1``), redrawn every ``permute_every_rows`` rows (default ``window_rows``, at
        least 1), in blocks of ``perm_block`` rows (default 1, from 1 to
        ``window_rows``). *The window test* in Notes gives the cost and the size.
    ``norm``
        ``"window"``: ``"l1"`` (the default) or ``"linf"``.
    ``seed``
        ``"window"``: seeds the permutation draws, at least 0, so two runs
        with the same seed report the same critical values. Default 0
        (``None`` is 0), and nothing draws from the system's entropy.
    ``change_action``
        ``"window"``: what a flag does, in the words ``drift_action`` takes:
        ``"flag"`` (the default) flags and carries on, and ``"reset"``
        empties the windows and starts over. A ``"monitor"`` span and a
        ``"sequential"`` cycle end at their flag already, so ``"reset"`` is
        refused there.
    ``monitor_rows``
        ``"sequential"``: the rows monitored after each history, the paper's
        ``floor(m T)``. Default ``window_rows`` (``T = 1``). At least 2.
    ``boundary_gamma``
        ``"sequential"``: the boundary's exponent, ``0 <= boundary_gamma <=
        0.49``. Default 0, the straight boundary ``1 + k/m``. Above 0 a
        change soon after the history is caught sooner and a late one later,
        at more false alarms (*Choosing boundary_gamma* in Notes). At 0.5 the
        boundary would be crossed with probability 1, and the solve behind
        the critical value costs time in proportion to ``1 / (1/2 -
        boundary_gamma)`` on the way there, so a value above 0.49 is refused.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. ``stat``, ``crit``, ``flag`` and ``since_flag``
    are null except on a row where a statistic is due, and
    ``since_change`` except on a flag. The statistic reported on a row
    includes that row, and is reported before the row is learned. So a
    zero-weight row is reported as if it would be learned. It then does not
    enter the span, the history or the monitoring period, does not advance
    ``since_flag`` for the rows after it, and if it flags, it resets neither
    ``since_flag`` nor, under ``change_action="reset"``, the windows. A
    clock break (a step ``gap_cap`` shortened, or a session change)
    abandons the span, the windows or the cycle.
    `docs/OUTPUTS.md#corrchange
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#corrchange>`_
    gives each field's type and when it is null:

    ``stat``
        The test statistic: for the span, for the pair of windows, or for the
        monitored row against its boundary.
    ``crit``
        The critical value it is compared against.
    ``flag``
        True on the row where ``stat`` crossed ``crit``.
    ``since_flag``
        Learned rows since the last flag.
    ``since_change``
        On a flag, the rows since the change it dates, counted through the
        flag's row from the first changed one. That row is the one after the
        maximum of ``j * |rho_j - rho_m|`` over the span (``"monitor"``), the
        one the paper's Eq. 8 gives (``"sequential"``), or the second
        window's first (``"window"``). Null otherwise. In the pairwise test
        (without ``scalar``), it is null on a ``"sequential"`` flag that has
        fewer than two monitored rows before it. And it is null where either
        feature of the flagged pair is constant over those rows, since a
        correlation then has no spread to divide by.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row, null where nothing decays.
    ``withheld_reason``
        Why the row's fields were withheld: null on every row at the default
        ``min_weight``, and ``below_min_weight`` while ``weight_sum`` is
        below a ``min_weight`` given.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), ``TypeError`` for
    ``targets``, which this model does not take, and ``ValueError`` naming
    the problem for:

    - fewer than two features, and an ``alpha`` outside ``(0, 1)``;
    - a ``kind``, ``alpha_adjust``, ``norm`` or ``change_action`` not one
      of its words;
    - ``window_rows`` left out or below the kind's minimum;
    - a parameter of another kind, and ``half_life`` or ``lam`` without
      ``scalar``;
    - under ``"sequential"`` without ``crit``, a share of ``alpha`` below
      ``5e-11`` a pair;
    - a ``bandwidth`` of 0 or above ``window_rows``, a ``crit`` not finite
      or not above 0, and an ``n_perm``, ``permute_every_rows``,
      ``perm_block`` or ``seed`` outside the range its parameter entry
      gives, an ``n_perm``
      among them too small to flag at ``alpha``;
    - a ``boundary_gamma`` outside ``[0, 0.49]``, and a ``monitor_rows``
      below 2;
    - ``change_action="reset"`` under ``"monitor"`` or ``"sequential"``;
    - a switch of a diagnostic that reads residuals.

    .. rubric:: The update

    ``"monitor"`` is the closed-sample constancy test of Wied, Krämer &
    Dehling (2012), run over consecutive spans of ``window_rows`` rows. At
    the last row of a span, per pair, with ``m`` the span's rows and
    ``rho_j`` the sample correlation of its first ``j`` rows:

    .. code-block:: text

        Q = max_{2 <= j <= m} (j / sqrt(m)) * |rho_j - rho_m| / D

    ``D`` is the delta-method long-run standard deviation of ``rho``
    (*The long-run deviation* in Notes). Over the pairs the statistic is the
    maximum, and the level is ``alpha / npairs`` under the default
    Bonferroni adjustment, or ``alpha`` under ``alpha_adjust="none"``.

    ``"sequential"`` is the monitoring procedure of Wied & Galeano (2013). A
    cycle is ``window_rows`` learned rows of history (``m``), from which
    each pair's correlation ``rho_hist`` and its ``D`` are read as for
    ``"monitor"``. Then the ``k``-th monitored row, for ``k`` up to
    ``monitor_rows``, reports

    .. code-block:: text

        V_k  = (k / sqrt(m)) * (rho_mon_k - rho_hist) / D
        stat = max over pairs of |V_k| / w(k / m)
        w(b) = (1 + b) * (b / (1 + b)) ** boundary_gamma

    with ``rho_mon_k`` the correlation of the ``k`` monitored rows so far,
    this one included, and flags where ``stat`` passes ``crit``. The cycle
    ends at a flag or after its ``monitor_rows``-th row, and the next learned
    row starts a new history. Nothing is reported during the history, nor on
    the first monitored row, since a correlation needs two.

    ``"window"`` reports ``norm(vech(R_pre - R_post))`` over two adjacent
    blocks of ``window_rows`` rows, ``R_pre`` and ``R_post`` their
    correlation matrices: how big the change is. The monitor asks instead
    whether the span was constant. It flags against ``crit``, or without one against a
    permutation quantile (*The window test* in Notes).

    .. rubric:: Example

    This code tests two features whose correlation jumps from 0 to 0.8 at
    row 1,100, over spans of 200 rows, and prints the span that flagged and
    the row its change was dated to:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(9)
        a, e = rng.standard_normal(2_000), rng.standard_normal(2_000)
        rho = np.where(np.arange(2_000) < 1_100, 0.0, 0.8)   # the jump at row 1,100
        b = rho * a + np.sqrt(1 - rho**2) * e
        pl.DataFrame({"a": a, "b": b}).write_parquet("pair.parquet")

        c = po.spec.corrchange(
            "brk", features=["a", "b"],
            kind="monitor",                                     # constant in each span?
            window_rows=200,                                    # a verdict on each span's last row
        )
        out = pl.scan_parquet("pair.parquet").online.fit_predict([c]).collect()
        due = out.unnest("brk").with_row_index("row").filter(pl.col("stat").is_not_null())
        flags = due.filter("flag")
        print(f"{due.height} spans tested, {flags.height} flagged")
        for r in flags.iter_rows(named=True):
            first = r["row"] - r["since_change"] + 1          # the first changed row, as dated
            print(f"span ending on row {r['row']}: stat {r['stat']:.2f} against {r['crit']:.2f}")
            print(f"the change dated to row {first}")

    .. code-block:: text

        10 spans tested, 1 flagged
        span ending on row 1199: stat 2.52 against 1.36
        the change dated to row 1099

    .. rubric:: See also

    - :func:`bocpd`, a posterior over the age of the current regime.
    - :func:`deco`, an equicorrelation that moves with the stream.
    - `docs/REGIMES.md
      <https://github.com/hgilde/polars-online/blob/main/docs/REGIMES.md>`_,
      the size and power of these tests, measured against their papers.

    .. rubric:: Notes

    **The long-run deviation.** ``D`` takes the five raw moments ``(x², y²,
    x, y, xy)`` centered at their span means and their Bartlett long-run
    covariance at bandwidth ``L = floor(ln m)`` with the paper's kernel, lag
    ``l`` at weight ``1 - l/L`` (their Appendix A.1). It maps them to
    ``(var_x, var_y, cov)`` and then to ``rho``. It is computed on the span
    centered at its means and scaled by its standard deviations, as the
    paper's ``xi`` series, so the columns' level does not enter. A pair
    within rounding of ``|rho| = 1`` has no ``D`` and no verdict, as a
    constant column has none.

    **The monitor test.** Under the null ``Q`` converges to ``sup|B|``, a
    Brownian bridge, so the critical value is the Kolmogorov quantile,
    computed from the series, not pinned, and it reproduces the published
    1.3581 at 5%. A span reports on its last row only, and tests only its
    own rows. So a change is found at most ``window_rows`` rows late, and
    reliably only where it falls well inside a span. A jump in correlation
    from 0 to 0.8 at row 1,100 of spans of 200 rows was found on 20 of 20
    seeds. At rows 1,190 and 1,195, near the span's end, it was found on 0
    to 3 of 20 (each of three sets of 20 seeds).
    ``docs/REGIMES.md`` §2 and §3 measure its size and power against those
    tables, and ``tests/test_corrchange.py`` holds its size near the nominal
    level.

    **The sequential test.** The critical value is Wied & Galeano's: with
    ``T = monitor_rows / window_rows``, ``crit = (T / (1 + T)) ** (1/2 -
    boundary_gamma) * q``. ``q`` is the ``1 - alpha`` quantile of
    ``sup_{0 < s <= 1} |W(s)| / s ** boundary_gamma`` for a Brownian motion
    ``W`` (their Eq. 7), at ``alpha / npairs`` under Bonferroni. At
    ``boundary_gamma = 0`` it comes from that law's series, 2.2414 at 5%, so
    ``crit`` is 1.5849 at ``T = 1``. Above 0 there is no series and the
    paper simulates. Here the law is solved as the diffusion it is, without
    simulation error. The values are within 0.03 of the paper's Table 1 and
    above it in 11 of its 12 cells, since a simulation on a grid misses
    crossings between its points. ``crit`` replaces the value.

    On a flag, ``since_change`` dates the change as the paper's Eq. 8 does:
    ``k_hat = argmax_{j < tau} j * |rho_mon_j - rho_mon_{tau-1}|`` over the
    rows monitored before the flag, the history left out (they found it
    distorts the estimate). It reports ``tau - k_hat``, the rows from the
    first changed one through the flag's. They found it biased late for a
    change early in the period and early for one in its middle, both less as
    ``m`` and ``T`` grow.

    What it assumes, as the paper does: the history's correlations are
    constant (their Assumption 1, which ``kind = "monitor"`` over the same
    rows checks), the rows have finite fourth moments, and dependence fades
    (near-epoch dependence, which admits GARCH). Against ``"monitor"``, every
    row is tested and the baseline is fixed. So a change is flagged as soon
    as it is large enough, and a drift that ``"monitor"``'s spans would each
    absorb is measured against one level.

    **Choosing boundary_gamma.** Above 0 the boundary is lower at the start
    of the period, so a change soon after the history is caught sooner and a
    late one later. It also increases the size: the paper measured 0.050-0.077
    at ``boundary_gamma`` of 0, 0.047-0.087 at 0.25, and 0.106-0.174 at
    0.45, for a nominal 0.05 on GARCH pairs (their Table 2). Their summary:

    .. list-table::
       :header-rows: 1
       :widths: 20 80

       * - ``boundary_gamma``
         - when
       * - 0
         - the pair is watched for a long time and false alarms are to be
           avoided, or a change is not expected soon after the history
       * - 0.45
         - to catch a change soon after it as fast as possible, false alarms
           accepted
       * - 0.25
         - a compromise

    **The window test.** Without ``crit`` the critical value is a
    permutation quantile: ``n_perm`` draws of the pooled rows shuffled
    between the two windows, redrawn every ``permute_every_rows`` rows. A
    report flags when fewer than ``floor(alpha * (n_perm + 1))`` draws reach
    its statistic. That is the standard permutation test, which counts the
    observed statistic as one more draw. On rows the null makes
    exchangeable, its rank among the ``n_perm + 1`` is uniform. So a test of
    a report against draws of its own windows has a size of that count over
    ``n_perm + 1``, at most ``alpha``: at alpha 0.05, 0.0476 at 20 draws and
    0.0498 at the default 200. The size here is about that, not exactly it.
    The critical value is drawn from the windows after a report and held for
    ``permute_every_rows`` reports, so each later report is compared with
    draws of windows up to ``permute_every_rows - 1`` rows older than its
    own. Over 200 streams of iid rows at alpha 0.05 and 200 draws, with the
    value held for ``window_rows`` reports (the default), 4.8% to 5.2% of
    non-overlapping spans flagged at windows of 50 to 400 rows, each within
    one standard error (0.24%) of 5%. Fewer than ``ceil(1 / alpha) - 1``
    draws could never flag, so such an ``n_perm`` is refused: at alpha 0.01,
    99 draws at least.

    The draws are in blocks of ``perm_block``, so that serial dependence
    does not make the null too liberal. It is not a sign-flip null. Negating a whole row leaves
    every ``x x'`` and so every correlation matrix exactly where it was, so a
    sign-flip null has no spread at all. The quantile is drawn from the
    windows after a row is reported, so the first report once the windows
    fill has no critical value and cannot flag: ``crit`` is null there, and
    in force from the next row. A ``change_action="reset"`` empties the
    windows and the value in force with them, so the first report after the
    refill has none either. The flag rate per row is not ``alpha`` here: two
    windows that slide by one row are almost the same windows, so a
    statistic above the quantile stays above it for a run of rows.

    A redraw shuffles the ``2 * window_rows`` pooled rows ``n_perm`` times
    and reads both windows' correlations each time, so it costs ``n_perm *
    window_rows * p**2`` for ``p`` features. At the default
    ``permute_every_rows`` (``window_rows``), a redraw comes once a window's
    length of rows, that is ``n_perm * p**2`` a row, whatever the window, so
    the draws' cost per row does not grow with ``window_rows``. Each drawn
    value is held for those ``window_rows`` rows. A smaller
    ``permute_every_rows`` redraws more often, at more cost a row.
    """
    model: dict[str, Any] = {
        "type": "corrchange",
        "kind": kind,
        "window_rows": window_rows,
        "alpha": alpha,
        "alpha_adjust": alpha_adjust,
        "bandwidth": bandwidth,
        "scalar": scalar,
        "crit": crit,
        "n_perm": n_perm,
        "permute_every_rows": permute_every_rows,
        "perm_block": perm_block,
        "norm": norm,
        "seed": seed,
        "change_action": change_action,
        "monitor_rows": monitor_rows,
        "boundary_gamma": boundary_gamma,
    }
    targets = _mirror_target(name, "corrchange", features, common, "its test is of")
    return _common(name, model, targets=targets, features=features, **common)
