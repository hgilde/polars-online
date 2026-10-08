"""The changepoint detectors' builders, :func:`bocpd` and :func:`corrchange`:
a part of :mod:`polars_online._spec` moved to a file of its own, keeping
that one under ``tests/test_repo_hygiene.py``'s 250 KB cap for a source
file. :mod:`polars_online.spec` documents them with the rest.
"""

from __future__ import annotations

from typing import Any, Unpack

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
    emission: str = "diag",
    prior_mean: list[float] | None = None,
    prior_kappa: float | None = None,
    prior_nu: float | None = None,
    prior_scale: list[float] | None = None,
    robust_beta: float | None = None,
    prune_below: float | None = None,
    max_run: int | None = None,
    warm_rows: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Bayesian online changepoint detection (Adams & MacKay 2007): a posterior over
    how long the current regime has lasted.

    Every other detector here answers "has something changed?" with a
    statistic. This one keeps a distribution over the run length, the rows
    since the last break, so the answer carries the age of the regime with
    it. "We are 40 rows into a regime" is different information from
    "something broke". Not a regression: no targets, and ``half_life``/``lam``
    are refused, since the run-length posterior is what forgets and
    ``hazard`` is how fast.

    .. rubric:: The recursion

    Their Algorithm 1, in log space, with ``H = 1 / hazard`` and ``pi_r`` run
    ``r``'s posterior predictive for this row:

    .. code-block:: text

        growth:      P(r_t = r+1, x_1:t) = P(r_t-1 = r, x_1:t-1) pi_r (1 - H)
        changepoint: P(r_t = 0,   x_1:t) = sum_r P(r_t-1 = r, x_1:t-1) pi_r H

    ``H`` there is the chance of a break after the row. A ``hazard`` given as
    a duration ``tau`` is the expected time between breaks instead. The step
    of ``d`` into a row then carries the chance ``h = 1 - exp(-d / tau)``
    that a break fell inside it, which belongs before the row. So ``h`` of
    every run's mass moves to the empty run before the row is read, and the
    recursion above runs with ``H = 0``, since no time has passed since the
    row:

    .. code-block:: text

        before the row:  P(r = 0) += h * sum_{r >= 1} P(r),   P(r) *= 1 - h for r >= 1

    Slot ``r`` keeps the conjugate statistics of exactly the ``r`` rows that
    hypothesis says preceded this one in the run. So slot 0 holds none and
    its predictive is the prior's, which is what makes "a new run starts
    here" something the data can vote on. A row takes ``O(runs * d²)``, and
    the run vector would grow by one every row. So runs below
    ``prune_below`` of the mass are dropped, and ``max_run`` folds every
    longer run into the last kept one. That run takes their mass and keeps
    its own statistics, so ``max_run`` bounds how much history any run
    holds, not only the length of the vector. ``run_mode`` saturates one
    below it.

    A row of weight ``w`` teaches at ``w / w_bar``, ``w_bar`` the mean weight
    of the rows learned from, this one included. That is in its run's
    statistics and in the likelihood it passes the recursion, so a heavier
    row is more evidence of both kinds, and a constant multiple of every
    weight changes nothing. What the row reports is read as a row of the
    mean weight, which is all ``predict`` can know. ``prior_kappa`` and
    ``prior_nu`` are in rows of that mean weight.

    .. rubric:: Parameters

    ``hazard``
        How likely a break is, in one of two forms:

        .. list-table::
           :header-rows: 1
           :widths: 26 36 38

           * - ``hazard``
             - what it is
             - the chance of a break
           * - a number, 250 by default
             - the expected rows between changepoints, on any clock
             - ``1 / hazard`` after every row
           * - a duration, such as ``"1h"``
             - ``tau``, the expected time between changepoints, on a
               temporal clock
             - ``1 - exp(-d / tau)`` for the step of ``d`` into each row,
               before the row is read

        A duration is a clock parameter like ``half_life``. It is refused
        without a temporal clock, beside a clock parameter given as a plain
        number, and at 0 or below. ``d`` is the step decay reads, so a gap
        past ``gap_cap`` counts as the cap. A step of 0 has no chance of a
        break: a row that shares the previous row's stamp cannot begin a
        run. A row of weight 0 applies its step's chance, since time passed,
        and teaches nothing, so two steps with nothing learned between them
        read as one step of their sum. Under ``prune_below = 0`` each step of
        0 leaves one run of no mass among the runs kept, until ``max_run``
        folds it.
    ``hazard_col``
        Read a per-row hazard from a column instead, declared in the targets
        slot the way a weight is. The value at a row is the expected rows
        between changepoints as of that row: ``1 / value`` is the chance of a
        break after the row, between it and the next. A null or non-finite
        value falls back to ``hazard``; a value of 1 or less is an error
        naming the row. A duration ``hazard`` puts its chance before each
        row, so the two are refused together.
        :meth:`polars_online.ModelBank.predict` reads the column too.
    ``emission``
        The predictive each run keeps, all exact conjugate updates:

        .. list-table::
           :header-rows: 1
           :widths: 18 40 42

           * - ``emission``
             - the predictive
             - what it adds
           * - ``"diag"`` (the default)
             - a normal-inverse-gamma per feature
             -
           * - ``"gaussian"``
             - a normal-inverse-Wishart over all of them
             - the one that can see a break in the correlation with the
               marginals unchanged: in ``run_mode``, since ``p_change`` never
               moves, and late, a median of 121 rows after the break in
               ``docs/REGIMES.md`` §5
           * - ``"robust"``
             - each row weighted by ``(pi(x) / pi(mode)) ** robust_beta`` in
               what the run learns and in the message it passes on
             - a 20-sigma row is atypical under every run, every tempered
               likelihood is about 1, and nothing moves. Not free of the
               data's units: centre and scale the features for it

        Without ``"robust"`` that one row is a changepoint (``p_change``
        0.91), and the run it starts carries the outlier in its mean.
        ``"gaussian"`` and ``"diag"`` are free of the data's units: a change of
        units multiplies every run's predictive density by one constant, which
        the posterior's normalisation takes out. ``"robust"`` is not. Its
        tempered message ``pi ** w`` carries that constant as ``c ** w``, and
        ``w`` differs from run to run, so the runs' odds move with the units. At
        a level of ``1e4`` with a spread of ``1e-2``, ``p_change`` moved by up
        to 0.86 against the same stream at level 0 and spread 1. Centre and
        scale the features before a ``"robust"`` run.
    ``robust_beta``
        The tempering under ``"robust"``. A trade: a whole new regime is a
        run of individually forgiven rows, so above about 0.2 nothing is
        ever detected again. The default, 0.1, ignores the outlier and still
        dates a four-sigma shift to the right row.
    ``prior_mean``, ``prior_kappa``, ``prior_nu``, ``prior_scale``
        The conjugate prior:

        .. list-table::
           :header-rows: 1
           :widths: 20 50 30

           * - parameter
             - what it is
             - default
           * - ``prior_mean``
             - ``mu_0``
             - the first ``warm_rows`` rows' mean
           * - ``prior_kappa``
             - the weight of that mean in rows
             - 1.0
           * - ``prior_nu``
             - the degrees of freedom
             - the smallest integer that gives the prior's variance a mean:
               ``d + 2`` under ``"gaussian"``, whose inverse-Wishart needs
               ``nu > d + 1``, and 3 under ``"diag"`` and ``"robust"``, whose
               per-feature normal-inverse-gamma needs ``nu > 2`` whatever ``d``
           * - ``prior_scale``
             - the prior scale of the variance as a list, in the data's units:
               one positive number ``[s]`` for ``s`` times the identity, or the
               ``d * d`` entries of a symmetric positive-definite matrix, row by
               row
             - the first ``warm_rows`` rows' covariance, its diagonal under
               ``"diag"`` and ``"robust"``

        A prior in the data's units decides what counts as surprising: too
        wide and the model goes quiet, because no row is surprising under a
        predictive that wide, and too narrow or off-centre and every run
        that begins on a row has no density for it. So the defaults are the
        data's own. With ``prior_mean`` or ``prior_scale`` left out, the first
        ``warm_rows`` learned rows report null and are held. Their weighted
        mean and covariance (``numpy.cov``'s with ``aweights``, the sample
        covariance at equal weights) set what was left out, and then they are
        read, in order, as rows of a model that had that prior from the start.
        At the default ``prior_nu`` the prior's mean variance is then that
        covariance. A feature that does not move over those rows has no
        variance to give; it takes ``2**-52 * max(mean**2, 1)``, the spread of
        a value known to ``2**-26`` of its level, since a predictive of no
        spread has no density. Under ``"gaussian"`` a covariance that is not
        positive definite, two features that are one, gives its diagonal. With
        both given there is no warm-up. ``prior_nu`` and ``prior_scale`` are
        ``2a`` and ``2b`` in the gamma parametrisation, which is how Adams and
        MacKay give their own finance example (``a = 1``, ``b = 1e-4``, ``hazard
        = 250``).
    ``warm_rows``
        The learned rows held to set the prior from, where ``prior_mean`` or
        ``prior_scale`` is left out. Default ``d + 2``, the smallest count whose
        covariance is positive definite in general, as :func:`hmm` and
        :func:`kmeans` name theirs. The prior is set once from those rows and
        kept for the stream's life, so at ``d + 2`` rows its scale rests on
        ``d + 1`` degrees of freedom. For one feature, a 3-row variance lies
        between 0.05 and 3 times the truth nine times in ten, and a warm-up
        that straddles a regime change sets a wide prior: a variance of 37
        across a 10-sigma step, against 0.53 from rows inside one regime.
        Where the first rows are not representative and rows are to spare,
        pass a ``warm_rows`` of tens, all from one regime. At least 2, and
        refused beside both priors, which leave nothing to set.
    ``prune_below``, ``max_run``
        The share of the mass below which a run is dropped (default
        ``1e-6``), and the run length every longer run is folded into
        (default 10,000). A changepoint collapses the runs to a few dozen,
        but a stream that does not break spreads them over thousands of run
        lengths. There ``max_run`` is the bound on the runs kept and on a
        row's work.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest.

    ``min_weight`` gates what is reported, never what is learned. A row
    whose predictive cannot be evaluated reports nulls, leaves the posterior
    where it stands, and is counted in
    :meth:`polars_online.ModelBank.solve_failures`.

    .. rubric:: Output

    One struct column named after the spec, all read before the row updates
    the posterior (`docs/OUTPUTS.md#bocpd
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#bocpd>`_):

    ``p_change``
        ``P(r_t <= 1)`` given this row: the alarm. Under a number ``hazard``
        it is ``P(r <= 1)`` and not ``P(r = 0)``: the changepoint branch and
        the growth branch share the same predictive, which makes the
        normalised mass at ``r = 0`` exactly ``H`` on every row whatever the
        data. Under a duration it is not "exactly ``H``" plus anything: the
        chance of a break is applied before the row, so nothing sits at ``r
        = 0`` after it, and ``p_change`` is the chance that this row began a
        run. A row whose step is 0 reports 0. On row one of a group ``P(r <=
        1)`` is 1 however the row looks: the default ``min_weight`` withholds
        the row, and at ``min_weight=0`` it reports that 1. A break the
        posterior accepts a row or two late shows as a fall in ``run_mode``,
        not in ``p_change``: a 5-sigma step at row 500 put ``run_mode`` at
        450 and 451 on rows 500 and 501, then 2, 3, 4 from row 502, and
        ``p_change`` never rose above 0.023.
    ``run_mode``
        The most likely run length before the row, so ``t - run_mode`` is
        the row the current run began on. This is the answer, and
        ``p_change`` is the alarm; they are not the same quality of signal.
        ``p_change`` is a per-row likelihood ratio, spiky and as big as the
        break is against the prior scale:

        .. list-table::
           :header-rows: 1
           :widths: 44 30 26

           * - the break
             - ``p_change``
             - the run length
           * - a tenfold variance step
             - 0.83 on the row itself
             - found within a few rows, dated to the right row
           * - a four-sigma mean shift under a diffuse prior
             - barely lifts
             - the same
           * - a change in correlation alone
             - never moves
             - the same
    ``run_mean``
        The posterior mean run length.
    ``pred_<f>``
        The pre-row predictive mean of each feature, mixed over runs.
    ``loglik``
        The row's log predictive density under that mixture.
    ``weight_sum``
        As everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where
        nothing decays, and why the row's fields are null where they are, as
        everywhere.

    .. rubric:: Example

    .. code-block:: python

        b = po.spec.bocpd(
            "regime", features=["ret"], group="stock_id",
            hazard=250.0,            # the expected run length
            prior_nu=2.0, prior_scale=[2e-4],   # 2a and 2b: Adams and MacKay's own finance example
            emission="diag",
            prune_below=1e-6,        # drop the runs thinner than this
        )
        out = po.ModelBank([b]).fit_predict(df).unnest("regime")
        # the row each group's run began on, counted in the group's own rows
        began = out.select(pl.int_range(pl.len()).over("stock_id") - pl.col("run_mode"))

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, ``ValueError`` for ``half_life`` or ``lam``, for a
    ``prior_scale`` that is neither ``[s]`` nor ``d * d`` entries, and for a
    duration ``hazard`` with no clock, beside a plain-number clock
    parameter, at 0 or below, or beside ``hazard_col``.
    """
    model: dict[str, Any] = {
        "type": "bocpd",
        "hazard": hazard,
        "hazard_col": hazard_col,
        "emission": emission,
        "prior_mean": prior_mean,
        "prior_kappa": prior_kappa,
        "prior_nu": prior_nu,
        "prior_scale": prior_scale,
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
    kind: str = "monitor",
    span_rows: int | None = None,
    alpha: float = 0.05,
    alpha_adjust: str = "bonferroni",
    bandwidth: int | None = None,
    scalar: bool = False,
    crit: float | None = None,
    n_perm: int | None = None,
    permute_every_rows: int | None = None,
    perm_block: int | None = None,
    norm: str = "l1",
    seed: int | None = None,
    reset_on_flag: bool = False,
    monitor_rows: int | None = None,
    boundary_gamma: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Has the correlation structure changed? Three tests, because there are
    three questions.

    ``kind = "monitor"`` asks whether the correlations were constant over a
    span of rows, with a published null. ``kind = "sequential"`` asks, as
    each row arrives, whether the correlations have left the level a stable
    history set. ``kind = "window"`` asks how big the change between two
    adjacent windows is, against a permutation quantile. Not a regression: no
    targets, no decay of anything but the optional standardiser, and nothing
    residual-based applies.

    .. rubric:: The tests

    ``"monitor"`` is the closed-sample constancy test of Wied, Krämer &
    Dehling (2012), run over consecutive spans of ``span_rows`` rows. At the
    last row of a span, per pair:

    .. code-block:: text

        Q = max_{2 <= j <= T} (j / sqrt(T)) * |rho_j - rho_T| / D

    ``rho_j`` is the sample correlation of the span's first ``j`` rows. ``D``
    is the delta-method long-run standard deviation of ``rho``. It takes the
    five raw moments ``(x², y², x, y, xy)`` centred at their span means and
    their Bartlett long-run covariance at bandwidth ``γ = floor(ln T)`` with
    the paper's kernel, lag ``l`` at weight ``1 - l/γ`` (their Appendix A.1),
    and maps them to ``(var_x, var_y, cov)`` and then to ``rho``. It is
    computed on the span centred at its means and scaled by its standard
    deviations, as the paper's ``xi`` series, so the columns' level does not
    enter. A pair within rounding of ``|rho| = 1`` has no ``D`` and no
    verdict, as a constant column has none. Under the null ``Q`` converges to
    ``sup|B|``, a Brownian bridge, so the critical value is the Kolmogorov
    quantile -- computed from the series, not pinned, and it reproduces the
    published 1.3581 at 5%. Over the pairs the statistic is the maximum and
    the level is ``alpha / npairs``. A span reports on its last row only, so
    a change is found at most ``span_rows`` rows late, against a null with
    published tables. ``docs/REGIMES.md`` §2 and §3 measure its size and
    power against those tables, and ``tests/test_corrchange.py`` holds its
    size near the nominal level.

    ``"sequential"`` is the monitoring procedure of Wied & Galeano (2013): a
    history taken as stable, then every row of a monitoring period tested
    against it as it arrives. A cycle is ``span_rows`` learned rows of
    history (``m``), from which each pair's correlation ``rho_hist`` and its
    long-run standard deviation ``D`` are read with the estimator
    ``"monitor"`` uses (bandwidth ``floor(ln m)``). Then the ``k``-th
    monitored row, for ``k`` up to ``monitor_rows``, reports

    .. code-block:: text

        V_k  = (k / sqrt(m)) * (rho_mon_k - rho_hist) / D
        stat = max over pairs of |V_k| / w(k / m)
        w(b) = (1 + b) * (b / (1 + b)) ** boundary_gamma

    with ``rho_mon_k`` the correlation of the ``k`` monitored rows so far,
    this one included, and flags where ``stat`` passes ``crit``. The cycle
    ends at a flag or after its ``monitor_rows``-th row, and the next learned
    row starts a new history. Nothing is reported during the history, nor on
    the first monitored row, since a correlation needs two.

    The critical value is Wied & Galeano's: with ``T = monitor_rows /
    span_rows``, ``crit = (T / (1 + T)) ** (1/2 - boundary_gamma) * q``.
    ``q`` is the ``1 - alpha`` quantile of ``sup_{0 < s <= 1} |W(s)| / s **
    boundary_gamma`` for a Brownian motion ``W`` (their Eq. 7; ``alpha /
    npairs`` under Bonferroni). At ``boundary_gamma = 0`` it comes from that
    law's series, 2.2414 at 5%, so ``crit`` is 1.5849 at ``T = 1``. Above 0
    there is no series and the paper simulates; here the law is solved as
    the diffusion it is, without simulation error. They are within 0.03 of
    the paper's Table 1 and above it in 11 of its 12 cells, since a
    simulation on a grid misses crossings between its points. ``crit``
    replaces the value.

    ``boundary_gamma`` trades early detection for late. Above 0 the boundary
    is lower at the start of the period, so a change soon after the history
    is caught sooner and a late one later. It also raises the size: the
    paper measured 0.047-0.087 at ``boundary_gamma`` of 0 and 0.25, and
    0.106-0.174 at 0.45, for a nominal 0.05 on GARCH pairs (their Table 2).
    Their summary:

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

    On a flag, ``since_change`` dates the change as the paper's Eq. 8 does:
    ``k_hat = argmax_{j < tau} j * |rho_mon_j - rho_mon_{tau-1}|`` over the
    rows monitored before the flag, the history left out (they found it
    distorts the estimate). It reports ``tau - k_hat``, the rows from the
    first changed one through the flag's. They found it biased late for a
    change early in the period and early for one in its middle, both less as
    ``m`` and ``T`` grow.

    What it assumes, as the paper does: the history's correlations are
    constant (their Assumption 1; ``kind = "monitor"`` over the same rows
    checks it), the rows have finite fourth moments, and dependence fades
    (near-epoch dependence, which admits GARCH). Against ``"monitor"``, every
    row is tested and the baseline is fixed. So a change is flagged as soon
    as it is large enough, and a drift that ``"monitor"``'s spans would each
    absorb is measured against one level.

    ``"window"`` is ``norm(vech(R_pre - R_post))`` over two adjacent blocks of
    ``span_rows`` rows -- how big the change is, rather than whether the span
    was constant. ``crit`` is a fixed threshold. Without one the critical
    value is a permutation quantile: ``n_perm`` draws of the pooled rows
    shuffled between the two windows, redrawn every ``permute_every_rows`` rows.
    The draws are in blocks of ``perm_block``, so that serial dependence does
    not make the null too liberal. It is not a sign-flip null, which a first
    reading of the literature suggests. Negating a whole row leaves every ``x
    x'`` and so every correlation matrix exactly where it was, so a sign-flip
    null has no spread at all. The flag rate per row is not ``alpha`` here:
    two windows that slide by one row are almost the same windows, so a
    statistic above the quantile stays above it for a run of rows.

    .. rubric:: Parameters

    A parameter that belongs to another kind is refused, naming the kinds it
    applies to.

    ``kind``
        ``"monitor"`` (the default), ``"sequential"`` or ``"window"``.
    ``span_rows``
        Required by every kind: the span ``"monitor"`` tests (at least 8),
        the history ``"sequential"`` monitors against (at least 8), or each
        of ``"window"``'s two windows (at least 3).
    ``alpha``, ``alpha_adjust``
        The level (default 0.05) and, under ``"monitor"`` and
        ``"sequential"``, how it is spread over the pairs (``"bonferroni"``,
        the default: ``alpha / npairs``). Under ``"window"`` one permutation
        statistic covers every pair at once, so there is nothing to spread,
        and an ``alpha_adjust`` other than the default is refused. Under
        ``"sequential"`` without ``crit`` each pair's share is refused below
        ``5e-11``: the critical value is solved from the boundary's law, and
        in a smaller tail the solve's quantile is off by more than the 0.03
        Wied and Galeano's own table is good to.
    ``bandwidth``
        ``"monitor"`` and ``"sequential"``: overrides the Bartlett bandwidth,
        ``floor(ln T)`` or ``floor(ln span_rows)``. At 1 only lag 0 is left.
    ``scalar``
        ``"monitor"`` and ``"sequential"``: test the equicorrelation of the
        standardised row (:func:`deco`'s ``u``) instead of every pair -- its
        mean, with the Bartlett long-run standard deviation of ``u`` in place
        of ``D``: one statistic however many columns there are.
        ``half_life``/``lam`` parametrise that standardiser and are accepted
        only here; neither kind decays anything else, so they are refused
        otherwise.
    ``monitor_rows``
        ``"sequential"``: the rows monitored after each history, the paper's
        ``floor(m T)``; default ``span_rows`` (``T = 1``). At least 2.
    ``boundary_gamma``
        ``"sequential"``: the boundary's exponent, ``0 <= boundary_gamma <=
        0.49``; default 0, the straight boundary ``1 + k/m``. At 0.5 the
        boundary would be crossed with probability 1, and the solve behind the
        critical value costs time in proportion to ``1 / (1/2 -
        boundary_gamma)`` on the way there, so a value above 0.49 is refused.
    ``crit``
        ``"window"``: a fixed threshold, in place of the permutation
        quantile. ``"sequential"``: replaces Wied & Galeano's critical value.
    ``n_perm``, ``permute_every_rows``, ``perm_block``, ``seed``
        ``"window"``'s permutation quantile: ``n_perm`` (default 200, at most
        2^20) draws, redrawn every ``permute_every_rows`` rows (default 50), in blocks of
        ``perm_block`` rows (default 1). ``seed`` (default 0) seeds the
        draws, so two runs with the same seed report the same critical
        values.
    ``norm``
        ``"window"``: ``"l1"`` (the default) or ``"linf"``.
    ``reset_on_flag``
        ``"window"``: empty the windows at a flag. Default ``False``. A
        ``"monitor"`` span and a ``"sequential"`` cycle end at their flag
        already.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest.

    A row is always part of the statistic reported on it: the report comes
    before the update, which is what makes the flag out of sample. So a
    zero-weight row is reported as if it would be learned. It then does not
    enter the span, the history or the monitoring period, does not advance
    ``since_flag`` for the rows after it, and does not reset it if it flags.
    A capped clock gap or a session change abandons the span, the windows or
    the cycle.

    .. rubric:: Output

    One struct column named after the spec, all null except where a
    statistic is due (`docs/OUTPUTS.md#corrchange
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#corrchange>`_):

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
        flag's row from the first changed one: after the CUSUM's maximum in
        the span (``"monitor"``), by the paper's Eq. 8 (``"sequential"``), or
        the second window (``"window"``). Null otherwise.
    ``weight_sum``
        As everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where
        nothing decays, and why the row's fields are null where they are, as
        everywhere.

    .. rubric:: Example

    .. code-block:: python

        c = po.spec.corrchange(
            "break", features=["x0", "x1"],
            kind="monitor",      # the constancy test; "window": how big the change is
            span_rows=100,       # nothing is reported until a span closes
        )
        out = po.ModelBank([c]).fit_predict(df).unnest("break")
        due = out.filter(pl.col("stat").is_not_null()).select("t", "stat", "crit", "flag")
        watch = po.spec.corrchange(
            "watch", features=["x0", "x1"],
            kind="sequential",
            span_rows=500,       # 500 rows of history, then
            monitor_rows=1000,   # each of up to 1000 rows tested as it arrives (T = 2)
            boundary_gamma=0.25, # a lower boundary early in the period
        )

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``; ``ValueError`` for:

    - ``span_rows`` below the kind's minimum;
    - a parameter of another kind;
    - ``boundary_gamma`` outside ``[0, 0.49]``;
    - ``monitor_rows`` below 2;
    - ``half_life``/``lam`` without ``scalar``.
    """
    model: dict[str, Any] = {
        "type": "corrchange",
        "kind": kind,
        "span_rows": span_rows,
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
        "reset_on_flag": reset_on_flag,
        "monitor_rows": monitor_rows,
        "boundary_gamma": boundary_gamma,
    }
    targets = _mirror_target(name, "corrchange", features, common, "its test is of")
    return _common(name, model, targets=targets, features=features, **common)
