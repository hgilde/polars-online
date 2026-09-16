"""Spec builders: ``polars_online.spec.ewridge("m", targets=[...], ...)``.

A spec is a plain dict: JSON-able, and the same thing as a ``[[specs]]``
entry in the CLI's TOML. It names a model, the columns the model reads, and
the stream parameters every model shares. A builder assembles one, checks
it and returns it; :class:`polars_online.ModelBank`, the ``online``
namespaces and :func:`polars_online.run` take lists of them.

The builders are the documented way to write a spec. A dict written by hand
is checked the same way when it is used, but only a builder can tell a
misspelt keyword from a missing one at the call.

**The stream parameters**, shared by every model (the README's "How a bank
sees a stream"; ``docs/PLAN.md`` section 3):

- ``targets``, ``features``: column names. A target is what the model
  predicts; a feature is what it reads from the same row.
- ``add_intercept``: prepend a constant 1 to the features (default ``True``).
- ``clock``: a monotone numeric column; the row count when ``None``. It need
  not be a time: sort by a feature and clock on it, and ``halflife`` is a
  bandwidth in that feature's units, which makes the fit a local regression
  in it (the README's "A clock that is not time").
- ``halflife`` or ``lam``: the decay, in clock units. A list of halflives
  means one fit per value.
- ``max_dclock``: a ceiling on the clock delta. Required with ``clock``;
  ``inf`` for no ceiling. It caps the step between two rows a model learns
  from, so a run of skipped rows hands the row after it at most
  ``max_dclock`` of clock, however long the run. A step the ceiling cut,
  one row's or a run's, also breaks adjacency: ``ew_cov`` and ``marginal``
  clear their lagged co-moments there, because a lag counts rows, and the
  row before such a gap is not one row back from the row after it.
- ``on_clock_reset``: what a backwards clock means -- ``"max"``, ``"zero"``,
  ``"reset_state"`` or ``"error"``.
- ``session``, ``session_gap``: the clock delta to apply where the session
  column changes.
- ``weight``: a row-weight column.
- ``min_periods``: how much weight a model must have seen before it
  reports, so that it never reports a number it is not yet informed enough
  to give. In ``n_eff`` units, not rows; outputs are null until it is
  reached, and the model learns from every row either way. A list gives
  one threshold per target. Six models keep a weight per target:
  ``ewridge``, ``lasso``, ``kalman``, ``huber``, ``quantile`` and ``holt``.
  Each of them checks a target's threshold against that target's own
  weight, which is the rows it was present on, at their raw weights,
  decayed, and inside the window under one. The other models check the
  shared ``n_eff``, and the ``n_eff`` field is the shared weight in every
  model. So a target that is often null is gated on its own rows, and its
  first prediction comes later than the others'.
- ``coef_every``: how often to snapshot the coefficients.
- ``label_delay``: hold each row back from learning until the clock has
  moved this much further on.
- ``group``, ``group_close``: one state per key, and when a key is finished.
- the ``emit_*`` switches, ``conformal``, ``resid_quantiles`` and the other
  diagnostics. ``sigma`` is the EW standard deviation of a slot's
  out-of-sample residuals, read before the row. Its weight ages on every
  row the model sees, a row with no prediction or with weight 0 included,
  so it forgets across a gap as the clock says. Under a ``window`` it is
  the window's, as the fit is, and so is everything that reads it: the
  ``resid_z`` field, drift's scale, the conformal band, and the ranking
  ``emit_selected`` and ``emit_averaged`` take. ``emit_averaged`` weighs
  each slot by ``exp(-eta * (sigma2 / sigma2_best - 1))``, the slot's EW
  squared error as a ratio to the best slot's, so ``average_eta`` means
  the same thing whatever the target's units; ``inf`` is
  ``emit_selected``'s argmin, a tie shared.

Each builder's signature lists them with their defaults, and
:func:`polars_online.spec.output_index` says which struct fields a spec
produces.

**Errors.** A builder raises ``TypeError`` for a name that is not a str, a
keyword it has not got, or a value of the wrong shape, naming the parameter
and what it takes::

    spec "m": halflife must be a number or a list of numbers, got str '10'

It raises ``ValueError``, naming the spec and the parameter, for a value the
model refuses: a count below 0; ``NaN`` anywhere; ``inf`` where it means
nothing (it is allowed where it does -- ``halflife``, ``max_dclock``,
``min_periods``, ``session_gap``, ``average_eta`` and the model parameters
that say so);
neither ``halflife`` nor ``lam``; ``clock`` without ``max_dclock``; a column
listed twice, or as both target and feature; a level outside ``(0, 1)``; an
option not in the list the message gives; and each model's own rules.
A parameter whose switch is off is refused rather than ignored:
``drift_delta``, ``drift_threshold`` or ``drift_action = "reset"`` without
``emit_drift``, ``average_eta`` without ``emit_averaged``,
``resid_autocorr_lag`` without ``emit_autocorr``, ``long_halflife`` without
``session_shrink``, ``session_gap`` without ``session``, an
``on_clock_reset`` other than the default without ``clock``, and
``coef_every`` on a model that reports no coefficients. Names are checked
too: a feature set named twice, a column twice in one set, an empty set,
and a spec named ``""``, ``"spec"`` or ``"group"``, which the bank's tables
use for their own columns.

A spec that came back from a builder is valid. One edited afterwards is
checked again wherever it is used, and a key no spec has is refused there
rather than ignored. Every way in checks a spec the same way, so a spec
one of them refuses is refused by all of them:
:class:`polars_online.ModelBank`, :func:`output_fields`,
:func:`output_index`, :func:`coef_fields`, the expression form and a run
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
