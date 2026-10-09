"""Spec builders: ``polars_online.spec.ewridge("m", targets=[...], ...)``.

A spec is what a model bank runs: one model, the columns it reads, and the
parameters every model shares -- the clock, the decay, the groups, the weights
and the warm-up. It is a plain dict, JSON-able, and the same thing as a
``[[specs]]`` entry in the CLI's TOML. A builder assembles one, checks it and
returns it. :class:`polars_online.ModelBank`, the ``online`` namespaces and
the ``online`` command line take lists of them.

The builders are the documented way to write a spec. A dict written by hand is
checked the same way when it is used. Only a builder can tell a misspelt
keyword from a missing one at the call.

.. rubric:: The stream parameters

Every builder takes these after its own parameters. The README's *How a bank
sees a stream* is the guide to them. This is the reference.

``targets``, ``features``
    Column names. A target is what the model predicts. A feature is what it
    reads from the same row. A model with no target (``ew_cov``, ``kmeans``,
    ``micro``, ``deco``, ``bocpd``, ``corrchange``, ``hmm`` and ``rcov``)
    takes ``features`` alone. ``ew_class`` takes a ``label`` column in the
    target's place. ``holt`` takes no features. A target computed from its
    own row's columns, such as a return against the mid, is a column: make
    it with Polars' ``with_columns`` before the bank, as a log ratio
    ``(pl.col("price_5m") / pl.col("mid")).log()`` or a difference, which
    sit about zero where ``hit_rate`` takes its sign. A
    :func:`polars_online.target` is a column under a name of its own. A
    target may also be a window expression looking ahead
    (:mod:`polars_online.ops`), named by its alias:
    ``(po.rewm_mean("mid", half_life="10s", window_size="1m") -
    pl.col("mid")).alias("fwd")``. The bank resolves it when the row's window
    closes, and learns the row then, under the ``embargo``. ``fit_predict``
    asks the embargo to cover the longest forward ``window_size``. Under
    ``group`` it needs a ``clock``.
    :func:`polars_online.stream.with_windows` with ``like=`` the spec writes
    the same target as a column (docs/PLAN.md task 104).
``fit_intercept``
    Whether the fit has a level of its own: a constant 1 is put in front of
    the features. Default ``True``. Without it nothing is centred, and a fit
    through the origin is least squares through the origin.
``clock``
    Why: rows are not evenly spaced, and forgetting should follow the stream's
    own time. What: a column that never runs backwards within a group; the row
    count when ``None``. A ``Datetime``, ``Date`` or ``Duration`` column is a
    temporal clock, and the parameters measured in clock units are then
    durations (*Clock units*, below). A numeric column is a clock in its own
    units, and need not be a time. Sort by a feature and clock on it:
    ``half_life`` becomes a bandwidth in that feature's units, and the fit a
    local regression in it.
``half_life``, ``lam``
    Why: a stream drifts, so older rows should count less. What: the decay.
    ``half_life`` is the clock distance over which a row's weight halves;
    ``lam`` is the factor per clock unit, ``0.5 ** (1 / half_life)``. ``inf``
    forgets nothing. A list of half-lives fits one instance per value, side by
    side. Units: clock units.
``gap_cap``
    Why: a gap in the stream, a weekend or a feed that stopped, would
    otherwise forget everything at once. What: a ceiling on the clock step
    between two rows a model learns from, finite and above ``0``, and
    required with ``clock``. The ceiling also caps the step a run of skipped
    rows hands the row after them, however long the run. A step the ceiling
    cut is a break. ``ew_cov`` and ``marginal`` clear their lagged
    co-moments there, because a lag counts rows, and the row before such a
    gap is not one row back from the row after it. A break releases nothing
    an ``embargo`` holds (its entry, below). For no forgetting at all, set
    ``half_life = "inf"``. Units: clock units.
``restart_after_step_back``
    What a clock that runs backwards within a group means. Why: a step back
    can be a new start, such as a replayed day or a restarted feed. Or it can
    be a late row: a transposed pair, a row a minute late, two sources never
    merged. Only the caller knows which size is which. What: unset (the
    default), every step back is refused: the chunk is refused naming the
    row, and the bank is untouched. Given, a step back no larger than it is
    a late row, refused the same way. A larger one starts the model over;
    ``0`` starts over at every step back. Needs ``clock``. It guards
    learning: ``predict`` scores a row before the last learned clock against
    the state as it stands, either way. Units: clock units.
``session``, ``session_gap``
    Why: a stream in market-data-like sessions has boundaries where the clock
    stops measuring time -- overnight, over a weekend. What: ``session`` names
    a column whose value changes at such a boundary. ``session_gap`` is the
    clock step to apply there, finite and at least 0, and capped at
    ``gap_cap`` like any other step; ``"reset"`` starts the model over
    instead. Units: clock units.
``weight``
    A row-weight column. A row of weight 0 is scored, advances the clock and
    teaches nothing; a null weight skips the row. ``seqtest`` and ``rcov``
    count rows, and take no weight or only 0 and 1.
``min_weight``
    Why: a model should report only once it has seen enough data to have
    converged, so it never reports a number it is not yet informed enough to
    give. What: the weight a model must have seen before its outputs stop
    being null; the model learns from every row either way. A list gives one
    threshold per target. Ten models keep a weight per target: ``ewridge``,
    ``lasso``, ``kalman``, ``huber``, ``quantile``, ``holt``, ``sgd``, ``pa``,
    ``ftrl`` and ``rls``. Each checks a target's threshold against that
    target's own weight: the rows it was present on, at their raw weights,
    decayed, and inside the window under one. ``rls`` learns only from rows
    on which every target is present, so its targets' weights are equal. The
    other models check the shared ``weight_sum``. The ``weight_sum``
    field is the shared weight in every model. So a target that is often null
    is gated on its own rows, and its first prediction comes later than the
    others'. Units: ``weight_sum`` units, not rows. The default depends on
    the model:

    .. list-table::
       :header-rows: 1
       :widths: 45 55

       * - the default
         - models
       * - one per unknown: the features, and the intercept when there is
           one
         - ``lasso``, ``kalman``, ``huber``, ``quantile``, ``rls``, ``sgd``,
           ``pa``, ``ftrl``
       * - the feature count plus one (1 for ``holt``)
         - ``ew_cov``, ``ew_class``, ``kmeans``, ``micro``, ``holt``
       * - 3
         - ``marginal``, ``deco``
       * - 1
         - ``bocpd``
       * - 0, each having a gate of its own
         - ``ewridge``, ``seqtest``, ``rcov``, ``hmm``, ``corrchange``
``min_settled_frac``
    Why: a history shorter than the half-life may not represent the process,
    such as regimes or seasons the half-life was chosen to average across.
    What: predictions are withheld while ``settled_frac``, how far the decay
    window has filled toward its steady state (``1 - 2 ** (-T /
    half_life)``, ``T`` the decay time seen), is below it. A fraction in
    ``[0, 1)``: ``0.5`` waits one half-life, ``0.75`` two. Default ``0``,
    off, since a fit kept as weighted means is unbiased from its first row
    when the process is stationary. Needs a decay: a finite ``half_life``, or
    ``lam`` below 1.
``max_error_inflation``
    ``ewridge``, ``rls``, ``kalman`` and ``lasso``. Why: a fit from too
    little data adds its estimation error to every prediction's. What:
    predictions are withheld while ``sqrt(1 + estimation variance / noise)``
    is at or above it, the factor by which estimation error is expected to
    inflate the prediction error over the noise floor. Per model:

    .. code-block:: text

        ewridge   sqrt(1 + edf / n_kish)          default sqrt(2)
        rls       sqrt(1 + k_total / n_kish)      off unless set
        lasso     sqrt(1 + df / n_kish)           off unless set, per path point
        kalman    sqrt(1 + z' P⁻ z / R)           off unless set, per row

    ``edf`` is the effective degrees of freedom the last solve used, at most
    ``k_total`` (``rls``'s fading prior makes its bound conservative);
    ``df`` the active coefficients plus the intercept (Zou, Hastie and
    Tibshirani 2007); ``n_kish`` Kish's effective sample size. ``kalman``'s
    is exact and per row: ``P⁻`` the coefficients' covariance carried
    through the row's clock gap, ``R`` the noise, ``obs_var`` or the
    residual variance, so a row far from the design reads larger. A ratio
    above 1; ``inf`` switches it off. It reads Kish's count, so uneven
    weights withhold for longer, and adding a feature moves the gate with
    the model.
``emit_error_inflation``
    ``ewridge``, ``rls`` and ``kalman``: write ``error_inflation_<slot>``,
    the same factor for the row's own features (*What a spec writes*,
    below). Default ``False``. Not ``lasso``, which keeps no factor to read
    a row's leverage from.
``coef_every``, ``max_rows_between_coefs``
    How often the ``coef`` field is filled, as ``solve_every`` and
    ``max_rows_between_solves`` schedule a solve. ``coef_every`` writes a
    ``coef`` row once the clock has moved that far since the group's last
    one, or since the group's first row before the first. It is a number of
    the clock's units, a duration on a temporal clock (``"5m"``), or ``0``
    for every row. ``max_rows_between_coefs`` writes one after that many
    rows the group's stream accepts: a row whose features and weight are
    usable, rows of weight zero and rows with a null target included. With
    both, whichever comes first. The clock is the one the models are stepped
    on, so a gap capped at ``gap_cap`` counts as the cap. Without a clock
    column it is the row's number, the first row being 1: ``coef_every=10``
    writes the tenth row, the twentieth and so on. A reset of the clock
    starts the count over. Under either, the ``coef`` rows are the same
    however the stream is chunked. With neither, the default, ``coef`` is
    filled on **each group's** last row within every chunk: one row per
    group per chunk, not one per chunk. Only that schedule follows the
    chunking; every other field is chunk-invariant, so one chunk or a
    thousand gives the same numbers. Refused on a model that reports no
    coefficients. Units: clock units, and accepted rows.
``embargo``
    Why: a target that is a forward quantity over ``h`` clock units is not
    known at the row it sits on. Learning it there hands the model ``h`` of
    the future before it predicts the rows in between. What: each row is
    scored where it sits and held back from learning until the clock has
    moved this much further on. The residual, ``sigma``, the metrics, drift,
    the conformal band, ``weight_sum`` and ``min_weight`` all see it then.
    The delay counts the time that passed on the clock column, skipped rows
    included, not the capped step the models decay by. Where a session
    change restarts the clock it counts ``session_gap``. A row exactly the
    delay later releases a row, and the time is held exactly: in integer
    nanoseconds on a temporal clock, and by one subtraction of the two rows'
    values on a number clock, in integers on an integer clock. A break
    releases nothing early: its events
    wait with the row after it, and run when that row is learned. A reset
    drops the held rows. Units: clock units, or without a ``clock`` the
    group's rows, a skipped row included.
``group``, ``group_close``
    One model state per value of the ``group`` column. ``group_close`` says
    when a key is finished, so its state can be dropped. ``"monotone"``: the
    key column never decreases, so a key below the largest one fed is done.
    ``"session"``: a key's span ends where its ``session`` value changes. A
    finished group's accumulators are emitted through
    :meth:`polars_online.ModelBank.closed_groups`, which is what keeps a bank
    over an unbounded key space bounded. Without it every key ever seen stays
    in memory.
the diagnostics
    ``emit_sigma``, ``emit_zscore``, ``emit_selected``, ``emit_averaged``
    with ``average_eta``, ``emit_metrics``, ``conformal`` with
    ``conformal_rate`` (default 0.05), ``resid_quantiles``, ``emit_autocorr``
    with ``resid_autocorr_lag``, ``emit_drift`` with ``drift_delta``,
    ``drift_threshold`` and ``drift_action``, ``emit_calibration`` with
    ``calibration_half_life``, ``emit_breaks`` with ``breaks_half_life``,
    ``emit_robust_se`` with ``robust_se_half_life`` and ``robust_se_lags``,
    ``emit_specification`` with ``specification_half_life`` and
    ``ljung_box_lags``, ``emit_tails`` with ``tails_half_life``, and
    ``emit_clocks``. Each adds fields to the
    output, listed below. A model with no residual refuses them by name.
    A ``*_half_life`` beside a switch is that diagnostic's own memory, in
    clock units: when left out, four times the model instance's half-life
    for the calibration and the instance's half-life for the others; ``inf``
    is the run-once form, which forgets nothing.

``standardize``, which seven models take, defaults to ``False`` in
``ewridge``, ``huber``, ``quantile`` and ``sgd``, and to ``True`` in
``kalman``, ``kmeans`` and ``micro``. Each builder says what it scales.

.. rubric:: Clock units

A parameter measured in clock units is written one of two ways, and the
clock column decides which:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - the clock
     - a clock parameter is
     - for example
   * - none
     - a number of rows
     - ``half_life=500``
   * - a numeric column
     - a number of the column's own units
     - ``half_life=600`` on a clock in seconds
   * - a ``Datetime``, ``Date`` or ``Duration`` column
     - a duration
     - ``half_life=pl.duration(minutes=10)``

A duration is a polars expression that reads no column, such as
``pl.duration(minutes=10)``, or a :class:`datetime.timedelta`. It may also be
polars' duration text, such as ``"10m"`` or ``"1h30m"``: whole numbers of
``ns``, ``us``, ``ms``, ``s``, ``m``, ``h``, ``d`` or ``w``, added up in any
order, as polars reads them (``"30m1h"`` is ``"1h30m"``). A month, a quarter
and a year have no fixed length, so ``"1mo"``, ``"1q"`` and ``"1y"`` are
refused. The spec keeps a duration as text, the form the command line's TOML
takes. A ``half_life`` grid names its instances by it (``@h10m``). ``0`` and
``inf`` mean the same in every unit, so they may stay numbers beside
durations; the word ``"inf"`` (or ``"infinity"``, in any case) is kept as the
number.

The clock parameters are ``half_life``, ``gap_cap``, ``restart_after_step_back``,
``session_gap``, ``coef_every`` and ``embargo`` above, ``drift_threshold``
and each diagnostic's own ``*_half_life`` below, and in the models
``window_size``,
``window_every``, ``solve_every``, ``ew_cov``'s ``pca_every``, ``micro``'s
``prune_every`` and the model half-lives: ``long_half_life``,
``select_half_life``, ``coef_half_life``,
``revert_half_life`` and ``trend_half_life``.

One parameter takes either form, with a meaning for each: ``bocpd``'s
``hazard``. A number, finite and above 1, is the expected rows between
changepoints, on any clock.
It binds to no clock unit, so it stands beside durations on a temporal clock.
A duration is the expected time between changepoints. It is a clock
parameter like the rest, so it needs a temporal clock and refuses a plain
number beside it. :func:`polars_online.spec.bocpd` says when each form's
chance of a break applies.

Each mixture is refused, naming the column, the parameter and the fix: a
temporal clock with a clock parameter given as a plain number, a numeric clock
with a duration, and one spec that gives both. A rate per clock unit has no
duration form, so a temporal clock refuses one too: ``lam``, and ``kalman``'s
``q``. Give the half-life the rate stands for as a duration instead.

A temporal clock is read in its own integer nanoseconds, and a duration in
seconds, so the column's own unit never reaches the fit. The same instants as
``Datetime("ms")``, ``Datetime("us")`` or ``Datetime("ns")`` give the same
numbers. A model reads only the gap between consecutive rows. The bank takes
that gap in integer nanoseconds before it becomes seconds, so a nanosecond
timestamp keeps its nanoseconds whatever the stream's age. The previous row's
instant is part of the state, so a loaded bank goes on exactly. A clock must
lie between the years 1677 and 2262, the range nanoseconds in a 64-bit integer
cover. A ``Date`` or a coarse ``Datetime`` past that is refused, naming the
row. A time zone changes nothing, because a ``Datetime`` is stored in UTC, so
a change of clocks for summer time neither stretches nor folds the clock.
Where a clock quantity reaches an output, it is in seconds: ``holt``'s trend
is per second. :meth:`polars_online.ModelBank.summary`,
:meth:`polars_online.ModelBank.groups` and
:meth:`polars_online.ModelBank.closed_groups` give clock values as seconds
since 1970. A ``Time`` column is refused, because a time of day starts again
at midnight.

.. rubric:: What a spec writes

A bank adds one struct column per spec, named after the spec. Every field but
``coef``, ``support_coef`` and ``se_coef`` is computed from the state
*before* the row updates it, so a prediction is out-of-sample and a diagnostic
never sees the row it describes. Those three report the fit *after* the
row: ``coef`` on row *t* is the fit row *t + 1* is predicted with. Under an
``embargo`` the row's own update waits for the delay, so ``coef`` on row *t*
is the fit after the rows *t* released, and row *t + 1* is predicted with it
only when *t + 1* releases none. A regression writes, with ``<t>`` a target:

``pred_<t>``
    The prediction. Null until ``min_weight`` is reached, and on a row the
    model skipped (a null feature or weight).
``resid_<t>``
    ``y - pred``; null where the target is null.
``weight_sum``
    The accumulated weight before this row's update and before its own decay.
    ``0`` on a stream's first row. One behind the row count while nothing is
    forgotten. ``1 / (1 - lam**d)`` once forgetting balances arrival, for
    unit rows ``d`` clock units apart (``lam = 0.5 ** (1 / half_life)``). A
    weight, not a count of rows: at a half-life of 600 with rows 0.1 apart it
    settles near 8,657, and Kish's ``n_kish`` is the sample size.
``coef``
    The coefficients of the fit after the row's update, the ones the next
    row is predicted with (under an ``embargo``, as above), as one flat
    list: per (target, grid combination) slot in the order the ``pred``
    fields declare them, the intercept and then one entry per feature. Null
    on rows where it is not filled (``coef_every``). :func:`coef_index` maps
    each position to its term.
    :func:`coef_fields` names the column each becomes when the struct is
    unnested.
``settled_frac``
    How far the decay window had filled toward steady state before this row:
    ``1 - 2 ** (-T / half_life)`` with ``T`` the decay time the models have
    seen, so ``0.5`` at one half-life, ``0.75`` at two, whatever the row rate.
    Null where nothing decays. What ``min_settled_frac`` gates on
    (`docs/WARMUP-AND-CONVERGENCE.md
    <https://github.com/hgilde/polars-online/blob/main/docs/WARMUP-AND-CONVERGENCE.md>`_).
``withheld_reason``
    Why the row's predictions are null, as a categorical --
    ``below_min_settled_frac``, ``below_min_weight`` or
    ``above_max_error_inflation``, the first in that order that withheld
    anything -- and null where nothing was withheld. A categorical rather
    than a string, because a string column costs sixteen bytes a row even
    when every value is null.
``support_coef``
    On ``coef``'s rows, for ``ewridge``, ``huber`` and ``quantile``, and of the
    same fit after the row: each coefficient's data share,
    ``1 - ridge * (S^-1)_jj`` in ``[0, 1]`` with ``S`` the system the solve
    inverts (for the robust models, the Gram as the loss weighs the rows),
    laid out like ``coef`` -- how much of it the data determined rather than
    the ridge. A duplicated pair reads ``0.5`` each, a clean design ``1``, a
    column the standardiser dropped ``0``; the intercept is not a share and
    is null. Not ``lasso``, whose L1 penalty has no such matrix.

A model that is not a regression writes fields of its own, which its builder
describes. Per model, the fields of the plainest spec are listed in
`docs/OUTPUTS.md
<https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md>`_, and
:func:`output_fields` lists them for the exact spec you built.

A grid writes one set of fields per instance, suffixed. A grid is a list of
half-lives, a list of ``ridge`` values, ``feature_sets`` or a ``lasso_path``.
``lasso`` also writes the path point it has selected, once per instance.
``emit_selected`` and ``emit_averaged`` choose across every slot of a
target, so their fields take no suffix:

.. code-block:: text

    pred_{target}{combo}{instance}     combo    = ""             single ridge, no feature sets
    resid_{target}{combo}{instance}             | __r{ridge}      ridge grid
    sigma_{target}{combo}{instance}             | __{set}         feature sets, single ridge
    weight_sum{instance}                        | __{set}_r{ridge}
    settled_frac{instance}                      | __l{lambda}     lasso_path point
    withheld_reason{instance}
    coef{instance}                     instance = ""             single half-life
    support_coef{instance}                      | @h{half-life}    half-life grid (@h600, @h10m)
    se_coef{instance}                           emit_se_coef
    se_coef_hc0{instance}, se_coef_hac{instance}  emit_robust_se
    penalty_selected_{target}{instance}         lasso: the path point in force
    selected_{target}                           emit_selected: the chosen slot
    pred_{target}__selected                     emit_selected: its prediction
    pred_{target}__averaged                     emit_averaged

``<slot>`` below is a target with its suffix. :func:`output_index` gives every
field with the values its name encodes, so a field is reached without building
its name. Avoid ``__`` and ``@`` in target names and feature-set labels if you
parse the names downstream.

The diagnostics add, per slot:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - switch
     - fields
     - what they hold
   * - ``emit_sigma``
     - ``sigma_<slot>``
     - The EW standard deviation of the slot's out-of-sample residuals.
       Its weight ages on every row the model sees, a row with no
       prediction or with weight 0 included, so it forgets across a gap
       as the clock says. Under a ``window_size`` it is the window's, as the
       fit is, and so is everything below that reads it.
   * - ``emit_zscore``
     - ``zscore_<slot>``
     - ``resid / sigma``: how surprising the row was, in units of the
       model's own recent error.
   * - ``emit_error_inflation``
     - ``error_inflation_<slot>``
     - ``sqrt(1 + h(x))`` for *this* row, ``h(x)`` the estimation variance
       of its prediction over the noise. It says how much estimation error
       is expected to inflate this prediction's error over the noise floor.
       Large for a row leaning on a direction the data never showed.
       ``ewridge``: ``x' S^-1 x / n_kish``, the row's leverage against the
       factor the fit came from; ``rls``: ``z' A^-1 z * s2 / s1``, the same
       in the sum form it keeps; ``kalman``: ``z' P⁻ z / R``, exact. One
       triangular solve or quadratic form a row, which is why it is opt-in.
       ``ewridge``'s and ``rls``'s gates read the stream average, which is
       free; ``kalman``'s reads this value.
   * - ``emit_se_coef``
     - ``se_coef``
     - On ``coef``'s rows and laid out like it, each coefficient's standard
       error, in ``coef``'s own units (the intercept's included). ``ewridge``:
       ``sigma * sqrt(diag(T M T'))``, ``M = S^-1 / n_kish`` the
       coefficients' covariance over the noise and ``T`` the map ``coef`` is
       read out by; it leaves out the ridge's sandwich, so it errs large.
       ``rls``: ``sigma * sqrt(s2 / s1 * diag(A^-1))``, the same in sum form.
       ``kalman``: ``sqrt(diag(T P T'))``, its posterior, exact. ``sigma`` is
       the row's EW out-of-sample residual std, which during warm-up still
       carries the estimation error, ``sqrt(1 + h)`` too large, so the error
       errs large there too; null until there is one. A report, not a gate.
       Refused for ``lasso`` (post-selection), ``huber`` and ``quantile``
       (an M-estimator's covariance is a sandwich), and the gradient models
       (no second moment).
   * - ``emit_robust_se``
     - ``se_coef_hc0``, ``se_coef_hac``
     - Beside ``se_coef``, on ``coef``'s rows and laid out like it, the
       standard errors of the least-squares sandwich ``B^-1 M B^-1`` at
       ``robust_se_half_life``: ``B`` the EW Gram of ``(1, x)``, and ``M``
       White's ``sum(w**2 * e**2 * z z')`` for ``se_coef_hc0``, with Newey
       and West's lag products to ``robust_se_lags`` at Bartlett's weights
       ``1 - l / (L + 1)`` for ``se_coef_hac``, written when there is a
       lag. ``e`` is the row's out-of-sample residual, which carries its
       estimation error where the classical sandwich's in-sample residual
       does not: run once on 1,500 rows it read 0.9-1.3% above
       statsmodels' ``cov_type="HC0"`` and ``"HAC"``, and on 3,000 rows
       under a 5- to 20-row embargo 2-9% above. The lags default to twice the
       target's horizon in rows: ``embargo`` on a spec with no clock
       column, ``0`` otherwise. On a target summing the next ``h`` rows'
       shocks against a persistent feature, the coefficient's true spread
       was 2.1-2.3 times ``se_coef`` at ``h = 5`` and 3.1-4.2 times at
       ``h = 20`` (``sqrt(h)`` is 2.2 and 4.5), and ``se_coef_hac`` read
       83-97% of it at ``L = h`` and 90-104% at ``2h``. ``ewridge`` and
       ``rls`` only, the least-squares fits; the ridge is left out, as
       ``se_coef`` leaves it out.
   * - ``emit_clocks``
     - ``scored_clock``, ``learned_clock``
     - The row's own clock, and the clock of the newest row the models had
       learned from, at a positive weight, when the row was scored. Without
       a delay that is the previous learned row; under ``embargo``, the
       newest row whose delay had passed. Both in the clock column's own
       type, a ``Datetime`` exact to the nanosecond; with no clock column,
       the row's index in its group, every row counted. Null on a skipped
       row; ``learned_clock`` null before the first learned row and after a
       reset. One pair per spec, last in the struct. ``scored_clock`` less
       ``learned_clock`` is at least the delay on every row.
   * - ``emit_selected``
     - ``selected_<t>``, ``pred_<t>__selected``
     - The grid slot with the lowest EW out-of-sample squared error so
       far, and its prediction. Needs more than one slot per target. Each
       slot's error decays at its own instance's half-life: like for like
       within a ridge or feature-set grid. Across a half-life grid a short
       half-life is ranked on fewer, more recent rows than a long one.
       ``lasso``'s ``select_half_life`` ranks its path on one half-life.
   * - ``emit_averaged``
     - ``pred_<t>__averaged``
     - Every slot's prediction averaged with weights
       ``exp(-eta * (sigma2 / sigma2_best - 1))``: each slot's EW squared
       error as a ratio to the best slot's, each at its own half-life, as
       ``emit_selected``'s are. So ``average_eta`` (default 1) means the
       same thing whatever the target's units. ``inf`` is
       ``emit_selected``'s argmin, a tie shared. Hedges where selection
       commits. The weights come from each slot's EW *mean* error, so
       they stay bounded and do not sharpen as rows accumulate, as a
       weighting by summed losses (river's ``EWARegressor``) does. A slot
       with no prediction or no ``sigma`` on the row is left out of that
       row's average; the field is null only when every slot is.
   * - ``emit_metrics``
     - ``ic_<slot>``, ``r2_<slot>``, ``hit_rate_<slot>``
     - Exponentially weighted correlation of prediction with target,
       out-of-sample R² against the running mean, and the share of rows
       whose sign the prediction got right, a row whose target or
       prediction is exactly 0 (1 for a ratio target) left out, as
       ``po.eval.metrics`` leaves it out. On a logistic ``sgd`` or
       ``ftrl`` fit: the point-biserial correlation, the Brier skill score
       and accuracy at a 0.5 threshold, under the same names. On a Poisson
       ``sgd`` fit ``hit_rate`` is null: a positive rate against a count
       has no sign to hit.
   * - ``conformal``
     - ``lo_<slot>``, ``hi_<slot>``, ``coverage_<slot>``
     - An interval ``pred ± q`` at the asked coverage, and the coverage it
       has delivered. ``q`` is a tracked quantile of ``|resid|``: it grows
       by ``conformal_rate * sigma * coverage`` on a miss and shrinks by
       ``conformal_rate * sigma * (1 - coverage)`` on a hit. Each step is
       times the row's weight over the EW mean weight of every row with a
       residual the layer has seen, scored or not, ``w / w̄``, so the
       long-run coverage weighted by ``w / w̄`` is the number asked for
       whatever the residuals do, and the weights' scale does not reach
       it. With unequal weights the share of rows covered can sit on
       either side of it. Null until the first
       ``sigma`` exists. The step is taken once per scored row, so ``q``
       moves faster in clock time where rows are denser; the delivered
       coverage decays on the clock.
   * - ``resid_quantiles``
     - ``abs_resid_q<p>_<slot>``
     - The exponentially weighted quantile of ``|resid|`` at level ``p``,
       at the model's half-life, each residual at its row's weight as in
       ``sigma``: an interval that assumes no distribution. One decaying
       DDSketch per slot answers every level, within ``tanh(1/128)``
       (0.78%) of the exact weighted quantile, in buckets that span the
       residuals of the last few dozen half-lives. Null until the first
       residual.
   * - ``emit_autocorr``
     - ``autocorr_<slot>``
     - The EW correlation of each residual with the one
       ``resid_autocorr_lag`` scored residuals back (default 1, at most
       2^20), within a run of adjacent rows. A gap capped by ``gap_cap`` or a session
       change starts a new run, as it clears the models' lags; the weights
       decay on every row's clock. A residual stream should look like
       noise; a value away from zero says the model is missing something.
   * - ``emit_drift``
     - ``drift_<slot>``
     - True on the row a Page-Hinkley detector on ``|resid|`` finds a
       break. Each residual over ``sigma`` is compared with its EW mean at
       the model's half-life, less ``drift_delta`` (default 0.5, in units
       of ``sigma``). The excess times each row's clock step is summed: a
       break when the sum climbs ``drift_threshold`` above its lowest
       point. So the threshold is in ``sigma`` times clock units, a clock
       parameter: a number of the clock column's units, or a duration on a
       temporal clock (``"20m"`` is one ``sigma`` of excess held for twenty
       minutes). It is required with a clock column, as ``gap_cap`` is;
       without one a row is one unit, and the default 20 is the classic
       test. How often it flags a stationary stream depends on the
       residuals' tails: at the defaults on a row clock, Gaussian residuals
       flagged no row in 600,000, and Student's t residuals with three
       degrees of freedom flagged 3 to 10 in every 200,000. The detector
       counts the same burst the same whatever the rows' density. Each
       residual is scored against the ``sigma`` before its row, which
       trails a moving scale further where rows are sparser.
       ``drift_action = "reset"`` also starts the model over there.
   * - ``emit_calibration``
     - ``calibration_slope_<slot>``, ``calibration_intercept_<slot>``,
       ``calibration_wald_<slot>``
     - Mincer and Zarnowitz's regression of the outcome on the
       out-of-sample prediction, ``y = a + b * pred``, exponentially
       weighted at ``calibration_half_life``, and Wald's statistic for
       ``a = 0, b = 1``:
       ``(n_kish - 2) * ((mean(y) - mean(pred))**2 + (b - 1)**2 * var(pred)) / s2``,
       ``s2`` the regression's residual mean square and ``n_kish`` Kish's
       size of its weights. The slope is the multiplier to put on a
       prediction. Run once (``inf``) with unit weights, ``wald / 2`` is
       the least-squares F statistic, ``F(2, n - 2)`` under Gaussian errors,
       and ``wald`` is ``chi2(2)`` as rows accrue: on calibrated fits it
       passed 5.99, its 5% value, in 5.0-7.0% of 400 streams. Beside a fit
       that forgets it is conservative, since the fit absorbs a
       miscalibration at its own pace, which is why its memory defaults to
       four times the model's half-life. On a fit whose slope was 0.7 it
       passed 5.99 on 5% of rows at the model's memory, on 42% at four
       times it and on 89% run once; on calibrated fits, on 0.3-0.6% of
       rows at the model's memory and 0-0.3% at four times. A run-once
       calibration keeps
       the first predictions for good, so give ``min_weight`` a few rows
       per coefficient: from ``k + 1`` rows they dominated it.
   * - ``emit_breaks``
     - ``studentized_<slot>``, ``cusum_<slot>``, ``cusum_sq_<slot>``,
       ``break_wald_<slot>``
     - Where the relationship broke. ``studentized`` is the row's
       recursive residual, ``resid / error_inflation``, over the spread of
       those before it at ``breaks_half_life``, read once that spread has 10
       rows of Kish's size; 1 stands in for the inflation on a model without
       one (all but ``ewridge``, ``rls`` and ``kalman``). ``cusum`` and
       ``cusum_sq`` are the EW sum of the studentized residuals before the
       row and the EW mean of their squares less 1, each standardized, so
       each is about ``N(0, 1)`` with no break. Run once (``inf``) they are
       Brown, Durbin and Evans' CUSUM and CUSUM of squares, ``cusum *
       sqrt(r)`` the path over ``r`` rows, read against ``±0.948 * (sqrt(T)
       + 2 r / sqrt(T))`` over a run of ``T`` (5%). With a memory they are
       moving sums. ``break_wald`` is the Wald distance between two
       least-squares fits of the target on the slot's features, one at the
       memory and one at four times it, about ``chi2(k)`` with no break and
       null run once. On 200 streams of 3,000 rows with a break at 1,500:
       run once, the CUSUM crossed on 3.5% with no break and on every
       stream whose intercept moved half a noise sd, 244 rows after it
       (statsmodels' own: 5.0% and 277); the CUSUM of squares on 5.0% and
       on every stream whose noise doubled. Neither sees a slope that moves
       on a centred feature, whose residuals keep a zero mean: the CUSUM
       crossed on 2.5%. Beside a fit at a half-life of 200, ``break_wald``
       passed ``chi2(3)``'s 0.01% value, 21.1, on no stream with no break
       and on every slope break, 94 rows after it; ``|cusum| > 3`` found
       every intercept break in 100 rows and ``|cusum_sq| > 3`` every
       variance break in 20. ``drift`` found 1.5% of the variance breaks
       there and none of the others.
   * - ``emit_specification``
     - ``ljung_box_<slot>``, ``breusch_pagan_<slot>``, ``reset_<slot>``
     - What the fit is missing, each at ``specification_half_life`` and
       Kish's size ``n``. ``ljung_box``: Ljung and Box's ``Q`` over
       ``ljung_box_lags`` lags of the residuals (default 10), ``chi2(L)``
       with nothing missing: a missing lag or a half-life too long. For a
       target that looks ahead ``h`` rows (``embargo`` on a spec with no
       clock column) it tests lags ``h`` to ``h + L - 1``, against
       Bartlett's covariance for autocorrelations past the residuals'
       built-in ``MA(h - 1)``; the plain ``Q`` passed its 5% value on 61% of
       the rows of five-row look-ahead streams with nothing missing, this one
       on 6.0-6.7%. ``breusch_pagan``: ``n R2`` of ``resid**2`` on the
       slot's features (Koenker's form), ``chi2(k)``: a spread that moves
       with them. ``reset``: Ramsey's test, the Lagrange multiplier for
       ``pred**2`` and ``pred**3`` beside ``pred`` in a regression of the
       residual, ``chi2(2)``: a curvature the fit is missing. Run once they
       are statsmodels' ``acorr_ljungbox``, ``het_breuschpagan`` and
       ``compare_lm_test`` on the out-of-sample residuals. On 200 streams of
       3,000 rows, each passed its 5% value on 2.0-7.0% of the rows of
       streams missing nothing, run once or windowed, and on 98-100% of
       those missing what it looks for: an AR(1) at 0.3, a spread linear in
       a feature, a square of one.
   * - ``emit_tails``
     - ``skew_<slot>``, ``kurtosis_<slot>``, ``jarque_bera_<slot>``
     - The EW skewness and excess kurtosis of the recursive residuals,
       ``resid / error_inflation`` (``resid`` on a model without one), at
       ``tails_half_life``, and Jarque and Bera's statistic, ``n / 6 *
       (skew**2 + kurtosis**2 / 4)`` at Kish's size, ``chi2(2)`` for
       Gaussian residuals. Run once they are ``scipy.stats``' ``skew`` and
       ``kurtosis`` and statsmodels' ``jarque_bera``. On 200 streams it
       passed 5.99 on 4.7-5.0% of the rows of Gaussian residuals, run once
       or windowed, and on every row of Student's t with 5 degrees of
       freedom. The kurtosis is what ``drift_threshold`` is set against:
       on 40 streams of 25,000 rows at a half-life of 200, the default 20
       flagged no row where the kurtosis read under 4 (Gaussian, t with 10,
       6 and 5 degrees of freedom), 0.4 rows in 100,000 where it read 8.6
       (t with 4) and none at 30, and 6.3 where it read 36 (t with 3), 2.1
       at 30, 0.4 at 50 and none at 80.

.. rubric:: Errors

A builder raises ``TypeError`` for a name that is not a str, a keyword it has
not got, or a value of the wrong shape, naming the parameter and what it
takes::

    spec "m": max_rows_between_coefs must be an int, got float 1.5

It raises ``ValueError``, naming the spec, the parameter and the value, for a
value the model refuses:

- a count below 0, or past what the Rust side holds it in (``2^32 - 1`` for
  the ``max_rows_between_*`` caps, ``max_iter``, ``update_every_rows`` and
  ``split_merge_every_rows``; ``2^64 - 1`` for the other counts), or ``NaN``
  anywhere;
- a count that sizes memory before the first row past its ceiling: a lag
  or ``n_perm`` past 2^20, ``kmeans``' ``k`` past 2^16 or its warm-up buffer
  past 256 MiB, ``rcov``'s lagged products past 256 MiB;
- ``inf`` where it means nothing (it is allowed where it does --
  ``half_life``, ``min_weight``, ``average_eta``, each diagnostic's
  ``*_half_life`` and the model parameters that say so);
- neither ``half_life`` nor ``lam``;
- ``clock`` without ``gap_cap``, or a ``gap_cap`` of ``0``;
- ``emit_drift`` with a ``clock`` and no ``drift_threshold``;
- a window target beside ``group`` without ``clock``;
- a column listed twice, or as both target and feature;
- a level outside ``(0, 1)``, or an option not in the list the message
  gives;
- and each model's own rules.

A parameter whose switch is off is refused rather than ignored:

- ``drift_delta``, ``drift_threshold`` or ``drift_action = "reset"`` without
  ``emit_drift``;
- ``average_eta`` without ``emit_averaged``;
- ``resid_autocorr_lag`` without ``emit_autocorr``;
- a diagnostic's ``*_half_life``, or its lags (``robust_se_lags``,
  ``ljung_box_lags``), without its switch;
- ``long_half_life`` without ``session_shrink``;
- ``session_gap`` without ``session``;
- ``restart_after_step_back`` without ``clock``;
- ``sgd``'s ``huber_delta``, ``quantile`` or ``eps`` beside a loss that does
  not use it, and its ``power`` beside a schedule other than
  ``"inv_scaling"``;
- ``kmeans``' ``dead_frac`` above 0 beside ``split_merge = 0``;
- ``hmm``'s ``transition`` or ``transition_prior`` beside ``tvtp_coef``, and
  its ``warm_rows``, ``seed_rule`` or ``seed`` beside given ``means`` and
  ``covs``;
- ``rcov``'s ``jitter`` or ``max_bandwidth`` under a kind other than
  ``"kernel"``, and its ``theta`` under one other than ``"preavg"``;
- ``kalman``'s ``coef_half_life`` beside ``q``, which it would derive;
- and ``coef_every`` or ``max_rows_between_coefs`` on a model that reports
  no coefficients.

Names are checked too: a feature set named twice, a column twice in one set,
an empty set, an empty ``feature_sets`` or ``blocks``, an empty target name,
an empty ``mahal_quantiles``, and a spec named ``""``, ``"spec"`` or
``"group"``, which the bank's tables use for their own columns.

A spec that came back from a builder is valid. One edited afterwards is
checked again wherever it is used, and a key no spec has is refused there
rather than ignored. Every way in checks a spec the same way, so a spec one of
them refuses is refused by all of them. :class:`polars_online.ModelBank`,
:func:`output_fields`, :func:`output_index`, :func:`coef_fields` and a run
config each fill its defaults and build its models.
"""

from polars_online._spec import (
    audit,
    bocpd,
    coef_fields,
    coef_index,
    corrchange,
    deco,
    ew_class,
    ew_cov,
    ewridge,
    ftrl,
    hmm,
    holt,
    huber,
    kalman,
    kmeans,
    lasso,
    marginal,
    micro,
    output_fields,
    output_index,
    pa,
    quantile,
    rcov,
    rls,
    seqtest,
    sgd,
)

__all__ = [
    "audit",
    "ew_class",
    "ew_cov",
    "ewridge",
    "ftrl",
    "hmm",
    "holt",
    "huber",
    "kalman",
    "kmeans",
    "lasso",
    "bocpd",
    "corrchange",
    "deco",
    "marginal",
    "rcov",
    "micro",
    "output_fields",
    "coef_fields",
    "coef_index",
    "output_index",
    "pa",
    "quantile",
    "rls",
    "seqtest",
    "sgd",
]
