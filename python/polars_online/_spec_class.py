"""The classifier's builder, :func:`ew_class`: a part of
:mod:`polars_online._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`polars_online.spec` documents it with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from polars_online._duration import Duration
from polars_online._kwargs import CommonKwargs
from polars_online._spec import _checked, _common
from polars_online._spec_words import quoted


@_checked
def ew_class(
    name: str,
    *,
    features: list[str],
    label: str,
    classes: list[str],
    covariance_type: Literal["full", "tied", "diag"] = "full",
    precision_prior: float,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    snapshot_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[Literal["thin", "refuse"], float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A Gaussian classifier on exponentially weighted moments, one set per class.

    Use it to predict a row's class, from a column of classes, by quadratic
    discriminant analysis, linear discriminant analysis or Gaussian naive
    Bayes (``covariance_type``). Use :func:`hmm` where no column gives the
    class and the model must infer a hidden regime, and :func:`kmeans` for
    clusters with no classes at all.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`polars_online.spec`. A ``label`` column takes the place of
    ``targets``. ``fit_intercept`` has nothing to act on, as there is no
    regression. No diagnostic that reads residuals applies, and each such
    switch is refused by name: ``emit_sigma``, ``emit_metrics``,
    ``conformal``, ``emit_drift`` and the rest. This model's own:

    ``label``
        The column that holds each row's class, read as a key: any dtype
        with a string form, so ``["0", "1"]`` for an integer column and
        ``["true", "false"]`` for a boolean one. On a row whose class is
        null, the model predicts the row and moves its clock, and no class
        learns from it. That is also how to predict rows whose
        class arrives late: null the class, keep the features.
    ``classes``
        Every value the label can hold, in the order the ``p_<class>`` fields
        are written. Required, at least two, each named once and none
        ``""``. A value it does not list is refused, with an error naming the
        row.
    ``covariance_type``
        The shape of each class's covariance, and the work a row takes, for
        ``k`` features:

        .. list-table::
           :header-rows: 1
           :widths: 20 36 44

           * - ``covariance_type``
             - the shape
             - per row
           * - ``"full"`` (the default)
             - a covariance per class (quadratic discriminant analysis)
             - each class keeps its Cholesky factor, and only the class a row
               moves is refactored: one ``k x k`` factorization per learned
               row. Under a ``window_size`` every class's covariance moves on
               every row, so every class is refactored
           * - ``"tied"``
             - the weight-averaged one, so the decision boundaries are linear
               (linear discriminant analysis)
             - one factorization
           * - ``"diag"``
             - the variances alone, which cannot see a correlation (Gaussian
               naive Bayes)
             - ``k`` numbers a class to keep and move on a row, where the
               other two keep ``k x k``
    ``precision_prior``
        A ridge on every class covariance, in the features' squared units,
        so that the first rows of a class, whose sample covariance is
        singular, are predicted with a finite covariance the same in every
        direction. Required, and finite and above 0: every row is predicted
        through each class's inverse covariance, where :func:`ew_cov` inverts
        only for ``partial_corr`` and ``mahal``. It washes out as the class
        fills in (*The prior scale* in Notes).
    ``window_size``, ``closed``
        A hard window on the rows each class's moments are computed from, in clock
        units, as for :func:`ewridge`: a row ``window_size`` old or older counts
        toward no class, and ``closed="both"`` keeps the one exactly that old. A
        hard window drops old rows entirely, so a classifier can follow class means
        that move: over a long history two regimes average together and the
        predicted classes go to chance. ``weight_sum`` is the weight inside the hard
        window, in the output and in the ``min_weight`` gate, and a row is predicted
        against the class moments of the rows inside the hard window.
    ``snapshot_every``, ``max_rows_between_snapshots``, ``window_budget``
        As for :func:`ewridge`: ``snapshot_every`` spaces the hard window's
        snapshots on the clock, in clock units, ``max_rows_between_snapshots``
        caps the rows between them, whichever comes first, and
        ``window_budget`` bounds each ring in MiB and thins or refuses past
        the bound. Each is refused without
        ``window_size``.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields, all but ``coef`` read before the row is learned.
    `docs/OUTPUTS.md#ew_class
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ew_class>`_
    gives each field's type and when it is null:

    ``class``
        The class with the largest posterior probability (the first, on a
        tie), an ``Enum`` of ``classes`` in their order. Null before
        ``min_weight`` and while no class has been seen.
    ``p_<class>``
        One per class in ``classes`` order: its posterior probability, so the
        ``p_`` fields sum to 1. Exactly 0 for a class no row has carried yet.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay, over every row it accepted, whether its class is
        null or not.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row, null where nothing decays.
    ``withheld_reason``
        Why the row's fields were withheld, null where they were not:
        ``below_min_weight`` until ``weight_sum`` reaches ``min_weight``.
        Before any row has carried a class, ``class`` is null, and so is the
        reason once ``weight_sum`` reaches ``min_weight``.
    ``coef``
        The class means after the row is learned, one per class in
        ``classes`` order, each one entry
        per feature. :func:`coef_index` lays the list out as ``(class,
        feature)``, and a class not yet seen is null.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), ``TypeError`` for
    ``targets``, since this model takes ``label``, and ``ValueError``
    naming the problem for:

    - fewer than two ``classes``, a class listed twice, or a class named
      ``""``;
    - a ``covariance_type`` other than ``"full"``, ``"tied"`` or ``"diag"``;
    - a ``precision_prior`` not finite or not above 0, and a
      ``window_size`` not finite or not above 0;
    - a ``label`` also listed as a feature;
    - a ``closed`` other than ``"right"`` or ``"both"``, a ``snapshot_every``
      below 0 or not finite, a ``max_rows_between_snapshots`` below 1, and a
      ``window_budget`` that is not one key, ``"thin"`` or ``"refuse"``, or
      whose MiB is not above 0;
    - ``closed="both"``, ``snapshot_every``, ``max_rows_between_snapshots``
      or ``window_budget`` without ``window_size``.

    And, from a call that learns, ``ValueError`` for a class not in
    ``classes``, naming the row.

    .. rubric:: The update

    Per row, ``w`` is the row's weight, ``λ = 0.5 ** (Δclock / half_life)`` its
    decay (``lam ** Δclock`` under ``lam``), and ``x`` its features. ``Δclock``
    is the clock's step since the last row the model did not skip, capped at
    ``gap_cap``, or ``session_gap`` at a session change. Each class ``c`` keeps
    a weight ``n_c``, a mean ``mu_c``, a centered covariance ``C_c`` and a prior
    scale ``s_c``. Before a row is learned, it is predicted against every class
    seen:

    .. code-block:: text

        pi_c = n_c / sum_c' n_c'
        omega_c = precision_prior * s_c
        M_c  = C_c + omega_c I                              ("full")
        M    = sum_c pi_c (C_c + omega_c I)                 ("tied")
        M_c  = diag(C_c) + omega_c I                        ("diag")
        l_c  = ln pi_c - 1/2 ln det M_c - 1/2 (x - mu_c)' M_c^-1 (x - mu_c)
        p_c  = exp(l_c - max_c' l_c') / sum_c' exp(l_c' - max l)

    Then the class of the row learns it:

    .. code-block:: text

        n_c   <- λ n_c + w
        mu_c  <- mu_c + (w / n_c) (x - mu_c)
        C_c   <- weighted Welford on (x - mu_c_old)(x - mu_c_new)'
        s_c   <- s_c * λ n_c_old / (λ n_c_old + w)
        n_c'  <- λ n_c'                                     every other class c'
        weight_sum <- λ weight_sum + w

    .. rubric:: Example

    This code learns two classes, ``a`` around ``(0, 0)`` and ``b`` around
    ``(2, 1)``, prints the class means found, and predicts a row at each
    planted mean:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        lab = rng.choice(["a", "b"], 4_000)
        x0 = np.where(lab == "a", 0.0, 2.0) + rng.standard_normal(4_000)
        x1 = np.where(lab == "a", 0.0, 1.0) + rng.standard_normal(4_000)
        pl.DataFrame({"x0": x0, "x1": x1, "lab": lab}).write_parquet("classes.parquet")

        cl = po.spec.ew_class(
            "cl", features=["x0", "x1"], half_life=float("inf"),
            label="lab", classes=["a", "b"],
            covariance_type="tied",               # one shared covariance: a linear boundary
            precision_prior=0.1,                  # lets a class be predicted from its first row
        )
        bank = po.ModelBank([cl])
        bank.fit(pl.scan_parquet("classes.parquet"))          # learn every row, keep no output
        means = bank.coef().pivot(on="term", index="class", values="coef")
        for r in means.iter_rows(named=True):
            print(f"class {r['class']}: mean ({r['x0']:.2f}, {r['x1']:.2f})")
        new = pl.DataFrame({"x0": [0.0, 2.0], "x1": [0.0, 1.0], "lab": [None, None]})
        for r in bank.predict(new).unnest("cl").iter_rows(named=True):
            print(f"at ({r['x0']:g}, {r['x1']:g}): class {r['class']}, p_b {r['p_b']:.2f}")

    .. code-block:: text

        class a: mean (0.01, -0.01)
        class b: mean (1.98, 1.00)
        at (0, 0): class a, p_b 0.08
        at (2, 1): class b, p_b 0.92

    .. rubric:: See also

    - :func:`ew_cov`, the moments each class keeps, for one set of rows.
    - :func:`hmm`, for a regime that no column names.
    - :func:`kmeans`, for clusters with no classes.

    .. rubric:: Notes

    **The other classes.** A row moves only its own class's mean,
    covariance and prior scale. The other classes' weights decay, which
    moves ``pi_c``. ``weight_sum`` counts every row the model accepted,
    whatever its class, so ``min_weight`` means the same number of rows as
    in every other model. A row with a null feature is skipped and learns
    nothing, and a row of weight 0 moves the clock alone.

    **The prior scale.** ``s_c`` starts at 1 and shrinks by
    ``λ * n_c / (λ * n_c + w)`` on every row the class learns, so the
    ridge washes out as the class fills in, exactly as :func:`ew_cov`'s
    ``precision_prior`` does.
    """
    model: dict[str, Any] = {
        "type": "ew_class",
        "classes": classes,
        "covariance_type": covariance_type,
        "precision_prior": precision_prior,
        "window_size": window_size,
        "closed": closed,
        "snapshot_every": snapshot_every,
        "max_rows_between_snapshots": max_rows_between_snapshots,
        "window_budget": window_budget,
    }
    if "targets" in common:
        msg = (
            f"spec {quoted(name)}: ew_class() takes no targets, since label names the "
            "column of its classes. Pass that column as label, and leave out targets."
        )
        raise TypeError(msg)
    return _common(name, model, targets=[label], features=features, **common)
