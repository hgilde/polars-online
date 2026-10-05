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
    target's place. ``holt`` takes no features. A target may also be a
    :func:`polars_online.target`, a column taken against another column of
    its own row: ``po.target("price_5m", relative_to="mid")``. The model then
    learns and predicts on that relative scale. Or a window expression
    looking ahead (:mod:`polars_online.ops`), named by its alias:
    ``(po.rewm_mean("mid", half_life="10s", window_size="1m") -
    pl.col("mid")).alias("fwd")``. The bank resolves it when the row's window
    closes, and learns the row then, under the ``embargo``. ``fit_predict``
    asks the embargo to cover the longest forward ``window_size``.
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
    others'. Units: ``weight_sum`` units, not rows.
``coef_every``
    How often the ``coef`` field is filled, in learned rows. ``0``, the
    default, fills it on **each group's** last row within every chunk only:
    one row per group per chunk, not one per chunk. Any value fills it there
    too. So ``coef``'s emission schedule follows the chunking, while every
    other field is chunk-invariant: one chunk or a thousand gives the same
    numbers. Refused on a model that reports no coefficients.
``embargo``
    Why: a target that is a forward quantity over ``h`` clock units is not
    known at the row it sits on. Learning it there hands the model ``h`` of
    the future before it predicts the rows in between. What: each row is
    scored where it sits and held back from learning until the clock has
    moved this much further on. The residual, ``sigma``, the metrics, drift,
    the conformal band, ``weight_sum`` and ``min_weight`` all see it then.
    The delay counts the time that passed on the clock column, skipped rows
    included, not the capped step the models decay by. Where a session
    change restarts the clock it counts ``session_gap``. A break releases
    nothing early: its events wait with the row after it, and run when that
    row is learned. A reset drops the held rows. Units: clock units, or
    without a ``clock`` the group's rows, a skipped row included.
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
    ``conformal_rate``, ``resid_quantiles``, ``emit_autocorr`` with
    ``resid_autocorr_lag``, and ``emit_drift`` with ``drift_delta``,
    ``drift_threshold`` and ``drift_action``, and ``emit_clocks``. Each adds
    fields to the output, listed below. A model with no residual refuses
    them by name.

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
``ns``, ``us``, ``ms``, ``s``, ``m``, ``h``, ``d`` or ``w``, largest first. A
month, a quarter and a year have no fixed length, so ``"1mo"``, ``"1q"`` and
``"1y"`` are refused. The spec keeps a duration as text, the form the command
line's TOML takes. A ``half_life`` grid names its instances by it (``@h10m``).
``0`` and ``inf`` mean the same in every unit, so they may stay numbers beside
durations.

The clock parameters are ``half_life``, ``gap_cap``, ``restart_after_step_back``,
``session_gap`` and ``embargo`` above, and in the models ``window_size``,
``solve_every`` and the model half-lives: ``long_half_life``,
``select_half_life``, ``coef_half_life``, ``revert_half_life``,
``level_half_life`` and ``trend_half_life``.

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

A bank adds one struct column per spec, named after the spec. Every field is
computed from the state *before* the row updates it, so a prediction is
out-of-sample and a diagnostic never sees the row it describes. A regression
writes, with ``<t>`` a target:

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
    The coefficients behind the fit, as one flat list: per (target, grid
    combination) slot in the order the ``pred`` fields declare them, the
    intercept and then one entry per feature. Null on rows where it is not
    filled (``coef_every``). :func:`coef_index` maps each position to its
    term. :func:`coef_fields` names the column each becomes when the struct
    is unnested.
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
    On ``coef``'s rows, for ``ewridge``: each coefficient's data share,
    ``1 - ridge * (S^-1)_jj`` in ``[0, 1]``, laid out like ``coef`` -- how
    much of it the data determined rather than the ridge. A duplicated pair
    reads ``0.5`` each, a clean design ``1``, a column the standardiser
    dropped ``0``; the intercept is not a share and is null.

A model that is not a regression writes fields of its own, which its builder
describes. Per model, the fields of the plainest spec are listed in
`docs/OUTPUTS.md
<https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md>`_, and
:func:`output_fields` lists them for the exact spec you built.

A grid writes one set of fields per instance, suffixed. A grid is a list of
half-lives, a list of ``ridge`` values, ``feature_sets`` or a ``lasso_path``:

.. code-block:: text

    pred_{target}{combo}{instance}     combo    = ""             single ridge, no feature sets
    resid_{target}{combo}{instance}             | __r{ridge}      ridge grid
    sigma_{target}{combo}{instance}             | __{set}         feature sets, single ridge
    weight_sum{instance}                             | __{set}_r{ridge}
    coef{instance}                     instance = ""             single half-life
                                                | @h{half-life}    half-life grid (@h600, @h10m)

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
     - ``sqrt(1 + h(x))`` for *this* row, ``h(x)`` its leverage against the
       factor the fit came from, over Kish's effective sample size. It says
       how much estimation error is expected to inflate this prediction's
       error over the noise floor. Large for a row leaning on a direction
       the data never showed. ``ewridge`` only; one triangular solve a row,
       which is why it is opt-in. The gate ``max_error_inflation`` reads the
       stream average, which is free.
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
       whose sign the prediction got right. On a logistic ``sgd`` or
       ``ftrl`` fit: the point-biserial correlation, the Brier skill score
       and accuracy at a 0.5 threshold, under the same names.
   * - ``conformal``
     - ``lo_<slot>``, ``hi_<slot>``, ``coverage_<slot>``
     - An interval ``pred ± q`` at the asked coverage, and the coverage it
       has delivered. ``q`` is a tracked quantile of ``|resid|``: it grows
       by ``conformal_rate * sigma * coverage`` on a miss and shrinks by
       ``conformal_rate * sigma * (1 - coverage)`` on a hit. Each step is
       times the row's weight over the scored rows' EW mean weight, so the
       long-run coverage is the number asked for whatever the residuals do,
       and the weights' scale does not reach it. Null until the first
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
       ``resid_autocorr_lag`` scored residuals back (default 1), within a
       run of adjacent rows. A gap capped by ``gap_cap`` or a session
       change starts a new run, as it clears the models' lags; the weights
       decay on every row's clock. A residual stream should look like
       noise; a value away from zero says the model is missing something.
   * - ``emit_drift``
     - ``drift_<slot>``
     - True on the row a Page-Hinkley detector on ``|resid|`` finds a
       break. Each residual over ``sigma`` is compared with its EW mean at
       the model's half-life, less ``drift_delta`` (default 0.5, in units
       of ``sigma``). The excess is integrated over the clock: a break when
       it climbs ``drift_threshold`` (default 20, in ``sigma`` times clock
       units) above its lowest point. The detector counts the same burst
       the same whatever the rows' density. Each residual is scored against
       the ``sigma`` before its row, which trails a moving scale further
       where rows are sparser. With rows one unit apart and no decay it is
       the classic test. ``drift_action = "reset"`` also starts the model
       over there.

.. rubric:: Errors

A builder raises ``TypeError`` for a name that is not a str, a keyword it has
not got, or a value of the wrong shape, naming the parameter and what it
takes::

    spec "m": half-life must be a number or a list of numbers, got str '10'

It raises ``ValueError``, naming the spec and the parameter, for a value the
model refuses:

- a count below 0, or ``NaN`` anywhere;
- ``inf`` where it means nothing (it is allowed where it does --
  ``half_life``, ``min_weight``, ``average_eta`` and the model parameters
  that say so);
- neither ``half_life`` nor ``lam``;
- ``clock`` without ``gap_cap``, or a ``gap_cap`` of ``0``;
- a column listed twice, or as both target and feature;
- a level outside ``(0, 1)``, or an option not in the list the message
  gives;
- and each model's own rules.

A parameter whose switch is off is refused rather than ignored:

- ``drift_delta``, ``drift_threshold`` or ``drift_action = "reset"`` without
  ``emit_drift``;
- ``average_eta`` without ``emit_averaged``;
- ``resid_autocorr_lag`` without ``emit_autocorr``;
- ``long_half_life`` without ``session_shrink``;
- ``session_gap`` without ``session``;
- ``restart_after_step_back`` without ``clock``;
- and ``coef_every`` on a model that reports no coefficients.

Names are checked too: a feature set named twice, a column twice in one set,
an empty set, and a spec named ``""``, ``"spec"`` or ``"group"``, which the
bank's tables use for their own columns.

A spec that came back from a builder is valid. One edited afterwards is
checked again wherever it is used, and a key no spec has is refused there
rather than ignored. Every way in checks a spec the same way, so a spec one of
them refuses is refused by all of them. :class:`polars_online.ModelBank`,
:func:`output_fields`, :func:`output_index`, :func:`coef_fields` and a run
config each fill its defaults and build its models.
"""

from polars_online._spec import (
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
