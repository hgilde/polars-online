"""Spec builders: ``polars_online.spec.ewridge("m", targets=[...], ...)``.

A spec is what a model bank runs: one model, the columns it reads, and the
parameters every model shares -- the clock, the decay, the groups, the weights
and the warm-up. It is a plain dict, JSON-able, and the same thing as a
``[[specs]]`` entry in the CLI's TOML. A builder assembles one, checks it and
returns it; :class:`polars_online.ModelBank`, the ``online`` namespaces and
:func:`polars_online.run` take lists of them.

The builders are the documented way to write a spec. A dict written by hand is
checked the same way when it is used, but only a builder can tell a misspelt
keyword from a missing one at the call.

.. rubric:: The stream parameters

Every builder takes these after its own parameters. The README's *How a bank
sees a stream* is the guide to them; this is the reference.

``targets``, ``features``
    Column names. A target is what the model predicts; a feature is what it
    reads from the same row. A model with no target (``ew_cov``, ``kmeans``,
    ``micro``, ``deco``, ``bocpd``, ``corrchange``, ``hmm`` and ``rcov``)
    takes ``features`` alone. ``ew_class`` takes a ``label`` column in the
    target's place, and ``holt`` takes no features.
``add_intercept``
    Whether the fit has a level of its own: a constant 1 is put in front of
    the features. Default ``True``. Without it nothing is centred, and a fit
    through the origin is least squares through the origin.
``clock``
    Why: rows are not evenly spaced, and forgetting should follow the stream's
    own time. What: a numeric column that never runs backwards within a group;
    the row count when ``None``. It need not be a time. Sort by a feature and
    clock on it, and ``halflife`` becomes a bandwidth in that feature's units,
    which makes the fit a local regression in it.
``halflife``, ``lam``
    Why: a stream drifts, so older rows should count less. What: the decay.
    ``halflife`` is the clock distance over which a row's weight halves;
    ``lam`` is the factor per clock unit, ``0.5 ** (1 / halflife)``. ``inf``
    forgets nothing. A list of halflives fits one instance per value, side by
    side. Units: clock units.
``max_dclock``
    Why: a gap in the stream, a weekend or a feed that stopped, would
    otherwise forget everything at once. What: a ceiling on the clock step
    between two rows a model learns from. Required with ``clock``; ``inf`` is
    no ceiling and ``0`` turns forgetting off. The ceiling also caps the step
    a run of skipped rows hands the row after them, however long the run. A
    step the ceiling cut breaks adjacency: ``ew_cov`` and ``marginal`` clear
    their lagged co-moments there, because a lag counts rows and the row
    before such a gap is not one row back from the row after it. Units: clock
    units.
``on_clock_reset``
    What a clock that runs backwards within a group means. ``"max"`` (the
    default): the step is ``max_dclock``. ``"zero"``: no step.
    ``"reset_state"``: the model starts over. ``"error"``: the chunk is
    refused, naming the row.
``session``, ``session_gap``
    Why: a stream in market-data-like sessions has boundaries where the clock
    stops measuring time -- overnight, over a weekend. What: ``session`` names
    a column whose value changes at such a boundary, and ``session_gap`` is
    the clock step to apply there, at most ``max_dclock``; ``"reset"`` starts
    the model over instead. Units: clock units.
``weight``
    A row-weight column. A row of weight 0 is scored, advances the clock and
    teaches nothing; a null weight skips the row. ``seqtest`` and ``rcov``
    count rows, and take no weight or only 0 and 1.
``min_periods``
    Why: a model should report only once it has seen enough data to have
    converged, so it never reports a number it is not yet informed enough to
    give. What: the weight a model must have seen before its outputs stop
    being null; the model learns from every row either way. A list gives one
    threshold per target. Six models keep a weight per target: ``ewridge``,
    ``lasso``, ``kalman``, ``huber``, ``quantile`` and ``holt``. Each checks a
    target's threshold against that target's own weight, which is the rows it
    was present on, at their raw weights, decayed, and inside the window under
    one. The other models check the shared ``n_eff``, and the ``n_eff`` field
    is the shared weight in every model. So a target that is often null is
    gated on its own rows, and its first prediction comes later than the
    others'. Units: ``n_eff`` units, not rows.
``coef_every``
    How often the ``coef`` field is filled, in learned rows. ``0``, the
    default, fills it on the last row of every chunk only; any value fills it
    there too. Refused on a model that reports no coefficients.
``label_delay``
    Why: a target that is a forward quantity over ``h`` clock units is not
    known at the row it sits on, and learning it there hands the model ``h``
    of the future before it predicts the rows in between. What: each row is
    scored where it sits and held back from learning until the clock has moved
    this much further on; the residual, ``sigma``, the metrics, drift, the
    conformal band, ``n_eff`` and ``min_periods`` all see it then. A reset
    drops the held rows; a session change releases them in order. Units: clock
    units, or rows without a ``clock``.
``group``, ``group_close``
    One model state per value of the ``group`` column. ``group_close`` says
    when a key is finished, so its state can be dropped: ``"monotone"`` (the
    key column never decreases, so a key below the largest one fed is done) or
    ``"session"`` (a key's span ends where its ``session`` value changes). A
    finished group's accumulators are emitted through
    :meth:`polars_online.ModelBank.closed_groups`, which is what keeps a bank
    over an unbounded key space bounded. Without it every key ever seen stays
    in memory.
the diagnostics
    ``emit_sigma``, ``emit_resid_z``, ``emit_selected``, ``emit_averaged``
    with ``average_eta``, ``emit_metrics``, ``conformal`` with
    ``conformal_rate``, ``resid_quantiles``, ``emit_autocorr`` with
    ``resid_autocorr_lag``, and ``emit_drift`` with ``drift_delta``,
    ``drift_threshold`` and ``drift_action``. Each adds fields to the output,
    listed below; a model with no residual refuses them by name.

.. rubric:: What a spec writes

A bank adds one struct column per spec, named after the spec. Every field is
computed from the state *before* the row updates it, so a prediction is
out-of-sample and a diagnostic never sees the row it describes. A regression
writes, with ``<t>`` a target:

``pred_<t>``
    The prediction. Null until ``min_periods`` is reached, and on a row the
    model skipped (a null feature or weight).
``resid_<t>``
    ``y - pred``; null where the target is null.
``n_eff``
    The accumulated weight before this row's update and before its own decay:
    ``0`` on a stream's first row, one behind the row count while nothing is
    forgotten, and ``1 / (1 - lam)`` once forgetting balances arrival.
``coef``
    The coefficients behind the fit, as one flat list: per (target, grid
    combination) slot in the order the ``pred`` fields declare them, the
    intercept and then one entry per feature. Null on rows where it is not
    filled (``coef_every``). :func:`coef_index` maps each position to its term
    and :func:`coef_fields` names the column each becomes when the struct is
    unnested.

A model that is not a regression writes fields of its own, which its builder
describes. Per model, the fields of the plainest spec are listed in
`docs/OUTPUTS.md
<https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md>`_, and
:func:`output_fields` lists them for the exact spec you built.

A grid writes one set of fields per instance, suffixed. A grid is a list of
halflives, a list of ``ridge`` values, ``feature_sets`` or a ``lasso_path``:

.. code-block:: text

    pred_{target}{combo}{instance}     combo    = ""             single ridge, no feature sets
    resid_{target}{combo}{instance}             | __r{ridge}      ridge grid
    sigma_{target}{combo}{instance}             | __{set}         feature sets, single ridge
    n_eff{instance}                             | __{set}_r{ridge}
    coef{instance}                     instance = ""             single halflife
                                                | @h{halflife}    halflife grid

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
       as the clock says. Under a ``window`` it is the window's, as the
       fit is, and so is everything below that reads it.
   * - ``emit_resid_z``
     - ``resid_z_<slot>``
     - ``resid / sigma``: how surprising the row was, in units of the
       model's own recent error.
   * - ``emit_selected``
     - ``selected_<t>``, ``pred_<t>__selected``
     - The grid slot with the lowest EW out-of-sample squared error so
       far, and its prediction. Needs more than one slot per target.
   * - ``emit_averaged``
     - ``pred_<t>__averaged``
     - Every slot's prediction averaged with weights
       ``exp(-eta * (sigma2 / sigma2_best - 1))``, each slot's EW squared
       error as a ratio to the best slot's, so ``average_eta`` (default
       1) means the same thing whatever the target's units; ``inf`` is
       ``emit_selected``'s argmin, a tie shared. Hedges where selection
       commits.
   * - ``emit_metrics``
     - ``ic_<slot>``, ``r2_<slot>``, ``hit_rate_<slot>``
     - Exponentially weighted correlation of prediction with target,
       out-of-sample R² against the running mean, and the share of rows
       whose sign the prediction got right. On a logistic ``sgd`` or
       ``ftrl`` fit: the point-biserial correlation, the Brier skill score
       and accuracy at a 0.5 threshold, under the same names.
   * - ``conformal``
     - ``lo_<slot>``, ``hi_<slot>``, ``coverage_<slot>``
     - An interval ``pred ± q`` at the asked coverage, ``q`` a tracked
       quantile of ``|resid|`` that grows by ``conformal_rate * sigma *
       coverage`` on a miss and shrinks by ``conformal_rate * sigma * (1 -
       coverage)`` on a hit, so its long-run coverage is the number asked
       for whatever the residuals do; and the coverage it has delivered.
       Null until the first ``sigma`` exists.
   * - ``resid_quantiles``
     - ``absresid_q<p>_<slot>``
     - A running quantile of ``|resid|`` per level ``p`` (the P² algorithm,
       five numbers per level): an interval that assumes no distribution.
   * - ``emit_autocorr``
     - ``autocorr_<slot>``
     - The EW correlation of each residual with the one
       ``resid_autocorr_lag`` rows back (default 1). A residual stream
       should look like noise; a value away from zero says the model is
       missing something.
   * - ``emit_drift``
     - ``drift_<slot>``
     - True on the row a Page-Hinkley detector on ``|resid|`` finds a
       break, at ``drift_delta`` tolerance (default 0.5, in units of the
       slot's ``sigma``) and ``drift_threshold`` (default 20).
       ``drift_action = "reset"`` also starts the model over there.

.. rubric:: Errors

A builder raises ``TypeError`` for a name that is not a str, a keyword it has
not got, or a value of the wrong shape, naming the parameter and what it
takes::

    spec "m": halflife must be a number or a list of numbers, got str '10'

It raises ``ValueError``, naming the spec and the parameter, for a value the
model refuses: a count below 0; ``NaN`` anywhere; ``inf`` where it means
nothing (it is allowed where it does -- ``halflife``, ``max_dclock``,
``min_periods``, ``session_gap``, ``average_eta`` and the model parameters
that say so); neither ``halflife`` nor ``lam``; ``clock`` without
``max_dclock``; a column listed twice, or as both target and feature; a level
outside ``(0, 1)``; an option not in the list the message gives; and each
model's own rules. A parameter whose switch is off is refused rather than
ignored: ``drift_delta``, ``drift_threshold`` or ``drift_action = "reset"``
without ``emit_drift``, ``average_eta`` without ``emit_averaged``,
``resid_autocorr_lag`` without ``emit_autocorr``, ``long_halflife`` without
``session_shrink``, ``session_gap`` without ``session``, an ``on_clock_reset``
other than the default without ``clock``, and ``coef_every`` on a model that
reports no coefficients. Names are checked too: a feature set named twice, a
column twice in one set, an empty set, and a spec named ``""``, ``"spec"`` or
``"group"``, which the bank's tables use for their own columns.

A spec that came back from a builder is valid. One edited afterwards is
checked again wherever it is used, and a key no spec has is refused there
rather than ignored. Every way in checks a spec the same way, so a spec one of
them refuses is refused by all of them: :class:`polars_online.ModelBank`,
:func:`output_fields`, :func:`output_index`, :func:`coef_fields`, the
expression form and a run config each fill its defaults and build its models.
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
