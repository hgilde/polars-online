"""The clustering builders, :func:`kmeans` and :func:`micro`: a part of
:mod:`polars_online._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`polars_online.spec` documents them with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from polars_online._duration import Duration
from polars_online._kwargs import CommonKwargs
from polars_online._spec import _checked, _common, _mirror_target


@_checked
def kmeans(
    name: str,
    *,
    features: list[str],
    n_clusters: int,
    warm_rows: int | None = None,
    init: Literal["first", "farthest", "k-means++", "lloyd"] | None = None,
    seed: int | None = None,
    update_every_rows: int | None = None,
    split_merge: float | None = None,
    split_merge_every_rows: int | None = None,
    dead_frac: float | None = None,
    standardize: bool = True,
    scale_floor: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted k-means over the feature columns.

    Use it to assign each row to one of a fixed number of clusters whose
    centers follow the stream, with the cluster read before the row is
    learned, so it is out-of-sample like every prediction. Use
    :func:`micro` where the number of clusters is not known, and
    :func:`hmm` where a row's cluster depends on the one before it.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`polars_online.spec`. This model takes ``features`` and no
    ``targets``. ``fit_intercept`` has nothing to act on, as there is no
    regression. No diagnostic that reads residuals applies, and each such
    switch is refused by name: ``emit_sigma``, ``emit_metrics``,
    ``conformal``, drift and the rest. This model's own:

    ``n_clusters``
        The number of centers. Required, at least 1, and at most 65,536
        (2^16): every row is compared with every center.
    ``warm_rows``, ``init``, ``seed``
        Seeding. The first ``warm_rows`` learned rows are held, at least
        ``n_clusters``: a smaller value is refused. Default 500, or
        ``n_clusters`` where that is more. The rows held are bounded at
        256 MiB, each row its values, its weight and 24 bytes besides (48
        bytes a row at two features): the refusal names the largest
        ``warm_rows`` that fits. Then the centers are placed by ``init``:

        .. list-table::
           :header-rows: 1
           :widths: 24 76

           * - ``init``
             - the centers
           * - ``"lloyd"`` (the default)
             - the best, by inertia, of ten ``"k-means++"`` starts, each
               followed by weighted Lloyd passes over the rows held until no
               row changes center
           * - ``"k-means++"``
             - greedy k-means++, scikit-learn's: each center after the first
               is the best of ``2 + floor(ln n_clusters)`` rows drawn by their
               squared distance from the centers so far, the one that leaves
               the least inertia
           * - ``"farthest"``
             - Gonzalez, from the first row
           * - ``"first"``
             - the first ``n_clusters`` distinct rows

        ``seed``, at least 0, drives the two random rules: the same seed
        gives the same centers. Default 0 (``None`` is 0), so a run draws the
        same centers every time. The rows held are replayed into the centers
        and freed, so from the row that completes seeding, memory grows with
        the model's state alone. Outputs are withheld until seeding and until ``weight_sum``
        reaches ``min_weight``. *Seeding* in Notes gives its timing and the
        rules at its edges.
    ``update_every_rows``
        How many learned rows the model gathers, each into its nearest
        center's batch, before it moves the centers by them, at least 1.
        Default 1, plain sequential k-means. Set it above 1 for mini-batch
        k-means: the centers then move once every that many rows.
    ``split_merge``, ``split_merge_every_rows``
        Move a center to a cluster that appears after seeding. Every
        ``split_merge_every_rows`` learned rows (default 100, at least 1),
        the two closest centers are merged where their distance is under
        ``split_merge`` times the sum of their radii and enough far rows
        have gathered. The freed center is placed at the mean of the far
        rows of the cluster that has gathered the most of them.
        ``split_merge`` is finite and at least 0, default 0.5. ``0`` turns
        off the merge and the far rows both, so every row is learned. *Far
        rows, split and merge* in Notes gives the rule.
    ``dead_frac``
        Re-place a center whose weight has decayed below ``dead_frac *
        weight_sum / n_clusters`` the same way, on whatever far rows there
        are. Default 0.05, and ``0`` disables it. At least 0 and below 0.5:
        at 0.5 the re-placed center starts at the floor itself. A center
        whose cluster vanished is re-placed ``log2(1 / dead_frac)``
        half-lives later (4.3 at the default, 2 at 0.25). The rule runs at a
        split-merge check, so beside ``split_merge = 0``, which runs none,
        the default is 0 and a value above 0 is refused. *Dead centers* in
        Notes gives the measurements.
    ``standardize``
        Measure distances in units of each feature's exponentially weighted
        standard deviation, tracked alongside the centers. The coordinates
        themselves are never rescaled, so the centers stay in the features'
        units. Default ``True``. ``False`` measures in the features' own
        units.
    ``scale_floor``
        Under ``standardize``, floor the metric's variance at this fraction
        of the feature's long-run variance, so a feature that goes quiet and
        then moves again is not infinitely far from every center. Default
        ``0.1``, and ``0`` is the exponentially weighted variance alone.
        Finite and at least 0. Refused beside ``standardize=False``, which
        reads no variance to floor. *The scale floor* in Notes gives the
        rule.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#kmeans
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#kmeans>`_
    gives each field's type and when it is null:

    ``cluster``
        The nearest center's index (``i64``), before the row is learned;
        null until seeding, and while the row is withheld.
    ``dist``, ``dist_second``
        The distance to that center, and to the runner-up, the second-nearest
        (null when ``n_clusters == 1``), so ``dist_second - dist`` is the
        margin. Under ``standardize`` it is in units of each feature's
        exponentially weighted (EW) standard deviation, and otherwise in the
        features' own units.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row, null where nothing decays.
    ``withheld_reason``
        Why the row's fields were withheld, null where they were not:
        ``below_min_settled_frac`` first of all, while ``settled_frac`` is
        below ``min_settled_frac`` where that is set, then
        ``below_min_weight`` on the first rows under the default
        ``min_weight``, then ``below_warm_rows`` on the rest of the rows held
        to seed the centers from.
    ``coef``
        The centers: ``n_clusters`` rows of ``len(features)``, flattened
        cluster-major. :func:`coef_index` lays it out, with ``cluster``
        reading ``0``, ``1``, ... and ``term`` the feature whose coordinate
        the position holds.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), ``TypeError`` for
    ``targets``, which this model does not take, and ``ValueError`` naming
    the problem for:

    - an empty ``features``;
    - an ``n_clusters`` of 0 or past 2^16, a ``warm_rows`` below
      ``n_clusters``, and held rows past 256 MiB;
    - an ``init`` not in the list, and a ``seed`` below 0;
    - an ``update_every_rows`` or ``split_merge_every_rows`` of 0;
    - a ``split_merge``, ``dead_frac`` or ``scale_floor`` below 0 or not
      finite, a ``dead_frac`` of 0.5 or more, a ``dead_frac`` above 0
      beside ``split_merge = 0``, and ``scale_floor`` beside
      ``standardize=False``;
    - a switch of a diagnostic that reads residuals.

    .. rubric:: The update

    Per row, ``w`` is the row's weight, ``λ = 0.5 ** (Δclock / half_life)``
    its decay (``lam ** Δclock`` under ``lam``), ``x`` the features, and
    ``c_j`` and ``n_j`` center ``j`` and its decayed weight. ``Δclock`` is
    the clock's step since the last row the model did not skip, capped at
    ``gap_cap``, or ``session_gap`` at a session change. Each center is the
    decayed weighted mean of the rows assigned to it (:func:`ew_cov`'s
    mean, per cluster):

    .. code-block:: text

        j*   = argmin_j |x - c_j|^2          in the units of dist
        n'_i = λ * n_i                                for every center i
        n'_j = λ * n_j + w
        c'_j = (λ * n_j * c_j + w * x) / n'_j         for the nearest j

    The distance is in units of each feature's EW standard deviation under
    ``standardize``, floored by ``scale_floor``, and in the features' own
    units without it. A far row (*Far rows, split and merge* in Notes)
    moves neither ``n_j`` nor ``c_j``.

    Alongside each center the model keeps its exponentially weighted squared
    radius ``r2_j``, the mean of ``|x - c_j|^2`` over the rows assigned
    there, which the split-merge rule reads. Rows are folded into
    per-center batches and applied every ``update_every_rows`` learned rows.

    .. rubric:: Example

    This code clusters rows drawn around two planted centers and prints
    each beside the center the model found:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(5)
        centers = np.array([[-2.0, 0.0], [3.0, 1.0]])       # the two planted centers
        pick = rng.integers(0, 2, 4_000)
        xy = centers[pick] + 0.5 * rng.standard_normal((4_000, 2))
        pl.DataFrame({"a": xy[:, 0], "b": xy[:, 1]}).write_parquet("blobs.parquet")

        km = po.spec.kmeans(
            "km", features=["a", "b"], half_life=float("inf"),
            n_clusters=2,
            warm_rows=100,                                    # seed the centers from 100 rows
        )
        bank = po.ModelBank([km])
        bank.fit(pl.scan_parquet("blobs.parquet"))           # learn every row, keep no output
        found = bank.coef().pivot(on="term", index="cluster", values="coef").sort("a")
        for (pa, pb), r in zip(centers, found.iter_rows(named=True)):
            print(f"planted ({pa:g}, {pb:g}): found ({r['a']:.2f}, {r['b']:.2f})")

    .. code-block:: text

        planted (-2, 0): found (-2.00, 0.02)
        planted (3, 1): found (3.02, 1.00)

    .. rubric:: See also

    - :func:`micro`, density clustering that finds the number of clusters.
    - :func:`hmm`, Gaussian states whose next state depends on the last.
    - :func:`ew_class`, Gaussian classes, one for each value of the column
      ``label=`` names.

    .. rubric:: Notes

    **Seeding.** Seeding under ``"lloyd"``, ten starts, runs once per
    group: 56 to 91 ms at 400 features and 8 centers, against 3.9 ms for
    ``"k-means++"``, its placement and the replay of the 500 rows. A
    ``"lloyd"`` start that is still moving rows after 300 passes stops there
    and is counted in :meth:`polars_online.ModelBank.solve_failures`; the
    starts measured on 500 rows needed at most 54. Under ``"first"``,
    seeding waits past ``warm_rows`` if it must for ``n_clusters`` distinct
    rows, holding up to ``max(warm_rows, 1000)`` rows.

    **Far rows, split and merge.** A row farther from its center than a
    blob of the typical radius produces (about four standard deviations of
    ``|x - c|^2`` above it) is far. It is predicted, but summarized instead
    of learned, so it neither drags the center nor widens the radius. At
    each check the two closest centers are compared, where there are at
    least three. If their distance is under ``split_merge`` times the sum of
    their radii, and enough far rows have gathered somewhere (at least
    three, and five percent of the weight learned since the last check),
    they are merged. The freed center is placed at the mean of the far
    rows of the cluster that has gathered the most of them. So a
    cluster that appears after seeding gets a center without a reset. Far
    rows still count in the radius at each check as if they sat at the cut,
    so a cut the data has outgrown widens until the rows are learned again.

    **Dead centers.** The freed center takes half the weight of the
    cluster it is split from, which keeps the other half, so from a cluster
    at an equal share both halves start at ``weight_sum / (2 n_clusters)``:
    the floor itself at 0.5. Each check then re-places one of them on the
    far rows, and the clustering is lost. Over 50 streams with a check every
    25 rows, a center whose blob vanished was re-placed once on every stream
    at 0.45 and below; 3 streams cascaded at 0.5, 29 at 0.6 and all 50 at
    0.9. A cluster lighter than ``dead_frac / n_clusters`` of the stream
    loses its center whenever any row is far.

    **The scale floor.** ``1 / var`` alone grows as ``2^Q`` over ``Q``
    half-lives of a feature going quiet (a flag that stops firing, a sensor
    at rest): a million at twenty, and the row on which the feature moves
    again is then infinitely far from every center. Floored, the weight
    grows as ``2^(Q/8) / scale_floor``, about 57 at twenty. The long-run
    variance is tracked at eight times the half-life, a feature at a time.
    Each row's weight and deviation are clipped against it, and its start
    is the medians of the feature's first five rows, so a row at the input
    bound moves it by a factor of 26 at most, which a few of its half-lives
    undo.
    """
    model: dict[str, Any] = {
        "type": "kmeans",
        "n_clusters": n_clusters,
        "warm_rows": warm_rows,
        "init": init,
        "seed": seed,
        "update_every_rows": update_every_rows,
        "split_merge": split_merge,
        "split_merge_every_rows": split_merge_every_rows,
        "dead_frac": dead_frac,
        "standardize": standardize,
        "scale_floor": scale_floor,
    }
    targets = _mirror_target(name, "kmeans", features, common, "its clusters are over")
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def micro(
    name: str,
    *,
    features: list[str],
    eps: float,
    beta_mu: float | None = None,
    max_clusters: int | None = None,
    prune_every: float | Duration | None = None,
    max_rows_between_prunes: int | None = None,
    macro_link: float | None = None,
    standardize: bool = True,
    scale_floor: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Density-based clustering over the feature columns, by small decayed summaries
    linked into clusters.

    Use it where the number of clusters is not known, their shapes are not
    round (moons, rings), rows that belong to no cluster should be reported,
    or clusters appear and vanish as the stream moves. Use :func:`kmeans`
    for a fixed number of clusters.

    .. rubric:: Parameters

    The model keeps a bounded set of small summaries (DenStream's
    micro-clusters), each a decayed weight, center and radius, and reads
    clusters off them as the chains of established summaries (those that
    have reached ``beta_mu``) whose centers lie within a link distance of
    each other.
    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`polars_online.spec`. This model takes ``features`` and no
    ``targets``. ``fit_intercept`` has nothing to act on, as there is no
    regression. No diagnostic that reads residuals applies, and each such
    switch is refused by name: ``emit_sigma``, ``emit_metrics``,
    ``conformal``, ``emit_drift`` and the rest. This model's own:

    ``eps``
        The bound on one summary's root-mean-square radius, per standardized
        coordinate (in the features' own units under ``standardize=False``):
        a summary takes a row only while its radius stays within
        ``eps * sqrt(p)`` for ``p`` features, so one value means the same at
        any ``p``. Required, finite and above 0. About 0.07 for
        two-dimensional shapes, 0.3 for well-separated Gaussians in twenty
        dimensions. Both ways to get it wrong show in the outputs:

        .. list-table::
           :header-rows: 1
           :widths: 34 36 30

           * - what you see
             - what it means
             - what to do
           * - nearly every row is an ``outlier`` and ``cluster`` stays null
             - ``eps`` is too small: no summary reaches ``beta_mu`` before it
               is pruned
             - increase ``eps``
           * - ``n_micro`` is about the number of clusters
             - ``eps`` is too coarse: the derived link, set from the
               summaries' spacing, then bridges them
             - lower ``eps``, or set ``macro_link=2``, which links only
               summaries that touch
    ``beta_mu``
        The weight at which a summary is established (DenStream's potential
        micro-cluster), in rows of the
        stream's mean weight. Default 3, finite and above 0. It is
        DenStream's point density (DBSCAN's ``MinPts``), so it is set against
        the arrival rate. With half-life ``h`` and ``v`` rows in each unit of
        the clock the stream's steady-state weight is about ``1.44 * v * h``.
        So a summary meant to hold a share ``s`` of the stream needs
        ``beta_mu`` of about ``1.44 * s * v * h``.
    ``max_clusters``
        The cap on live summaries, at least 1. Default 200.
    ``prune_every``, ``max_rows_between_prunes``
        A prune, which drops light summaries and links the rest, comes every
        ``prune_every`` clock units or every ``max_rows_between_prunes``
        learned rows (at least 1), whichever comes first, as
        ``solve_every`` and ``max_rows_between_solves`` schedule a
        regression's solve. With neither, every 100 learned rows.
        ``prune_every`` is finite and at least 0: a number of the clock's
        units, a duration on a temporal clock (``"10m"``), or ``0`` for every
        row. *The prune schedule* in Notes gives the rules.
    ``macro_link``
        The link distance ``L`` as a multiple of ``eps * sqrt(p)``: ``0``
        links nothing, so each summary is its own cluster,
        and ``2`` links summaries that touch. Finite and at least 0. Left
        out, ``L`` is derived at each prune from the spacing the
        summaries already show (*The link* in Notes).
    ``standardize``
        Measure in units of each feature's exponentially weighted standard
        deviation. Default ``True``. ``False`` measures in the features' own
        units.
    ``scale_floor``
        Under ``standardize``, floor the metric's variance at this fraction
        of the feature's long-run variance, so a feature that goes quiet and
        then moves again is not infinitely far from every center. Default
        ``0.1``, and ``0`` is the exponentially weighted variance alone.
        Finite and at least 0. Refused beside ``standardize=False``, which
        reads no variance to floor. *The scale floor* in Notes gives the
        rule.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields, all read before the row is learned.
    `docs/OUTPUTS.md#micro
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#micro>`_
    gives each field's type and when it is null:

    ``cluster``
        The label of the nearest established summary (``i64``), null while
        there is none.
    ``dist``
        The distance to that summary's center, in units of each feature's
        EW standard deviation under ``standardize``, and otherwise in the
        features' own units.
    ``micro_id``
        The id of the summary this row goes to (``i64``), the one it opens
        when none can take it.
    ``outlier``
        Whether no established summary takes it (``bool``).
    ``n_clusters``, ``n_micro``
        Live clusters and live summaries (``i64``), so churn is visible
        without comparing labels.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row, null where nothing decays.
    ``withheld_reason``
        Why the row's fields were withheld: ``below_min_settled_frac`` while
        ``settled_frac`` is below ``min_settled_frac``, where that is set,
        then ``below_min_weight`` while ``weight_sum`` is below
        ``min_weight``, and null after. A row before any summary is established is not withheld:
        its ``cluster`` and ``dist`` are null and ``outlier`` is true.
    ``coef``
        The established summaries, one ``[id, label, n, radius, c_1, ...,
        c_p]`` row each, flattened, as many rows as there are, so
        :func:`coef_index` does not apply.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), ``TypeError`` for
    ``targets``, which this model does not take, and ``ValueError`` naming
    the problem for:

    - an empty ``features``;
    - an ``eps`` or ``beta_mu`` not finite or not above 0;
    - a ``max_clusters`` or ``max_rows_between_prunes`` of 0;
    - a ``prune_every``, ``macro_link`` or ``scale_floor`` below 0 or not
      finite, and ``scale_floor`` beside ``standardize=False``;
    - a switch of a diagnostic that reads residuals.

    .. rubric:: The update

    A summary is ``(n, c, nu2)``: its decayed weight, its center, and the
    exponentially weighted (EW) mean squared distance of its rows from the
    center (DenStream's radius, with the fading function being the decay).
    Distances are measured in the metric ``mw_i = 1 / var_i`` under
    ``standardize``, ``var_i`` feature ``i``'s EW variance floored at
    ``scale_floor`` times its long-run variance (*The scale floor* in
    Notes), and ``mw_i = 1`` under ``standardize=False``. So ``eps`` is a
    bound per standardized coordinate (or per unit without
    ``standardize``), and the bound on a summary's squared radius in the
    metric is ``E = eps**2 * p``. ``x`` is the row's features, ``w`` the
    row's weight, ``w_bar`` the EW mean
    weight of the rows learned, and ``λ = 0.5 ** (Δclock / half_life)`` the
    row's decay (``lam ** Δclock`` under ``lam``). ``Δclock`` is the clock's
    step since the last row the model did not skip, capped at ``gap_cap``,
    or ``session_gap`` at a session change. Each row:

    .. code-block:: text

        n_i  <- λ n_i                                          every summary
        j    =  the nearest established summary, if r2(j, w_bar) <= E
                else the nearest summary not yet established, if r2(j, w_bar) <= E
                else none: a new summary opens at x with weight w and the next id
        r2(j, v) = a nu2_j + a b |x - c_j|^2,   a = n_j / (n_j + v),  b = v / (n_j + v)
        nu2_j <- min(r2(j, w), E)                              c_j the center before the row
        c_j  <- c_j + w / (n_j + w) * (x - c_j)
        n_j  <- n_j + w

    A summary is established once ``n >= beta_mu * w_bar``, and is never
    demoted: it stays established until a prune drops it. Established
    between two prunes, it takes the label of the nearest established
    summary within ``L``, else its own id. A new summary is
    opened at the cap ``max_clusters`` by evicting the lightest summary not
    yet established, else the lightest established one. At each prune the
    model drops the light summaries (*Pruning* in Notes), then links the
    established summaries by single linkage: centers within ``L`` of each
    other share a label.

    A row of weight ``w`` is admitted where a row of the mean weight would
    be and absorbed with its full weight, so a constant multiple of every
    weight moves nothing. A zero-weight row advances the clock and learns
    nothing.

    .. rubric:: Example

    This code clusters rows drawn around two planted centers, with a third
    that appears at row 3,000, and prints the clusters the model reads
    before and after it, and the share of rows it called outliers at the
    end:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(6)
        centers = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
        pick = rng.integers(0, 2, 6_000)                  # two blobs from the start
        pick[3_000:] = rng.integers(0, 3, 3_000)          # a third appears at row 3,000
        xy = centers[pick] + 0.3 * rng.standard_normal((6_000, 2))
        pl.DataFrame({"a": xy[:, 0], "b": xy[:, 1]}).write_parquet("blobs3.parquet")

        mc = po.spec.micro(
            "mc", features=["a", "b"], half_life=1_000.0,
            eps=0.1,                                        # the spread one summary holds
        )
        out = pl.scan_parquet("blobs3.parquet").online.fit_predict([mc]).collect()
        out = out.unnest("mc")
        for row in [2_999, 5_999]:
            print(f"row {row}: {out['n_clusters'][row]} clusters")
        print(f"outliers in the last 1,000 rows: {out['outlier'][-1_000:].mean():.1%}")

    .. code-block:: text

        row 2999: 2 clusters
        row 5999: 3 clusters
        outliers in the last 1,000 rows: 0.0%

    .. rubric:: See also

    - :func:`kmeans`, a fixed number of clusters whose centers follow the
      stream.
    - :func:`hmm`, Gaussian states whose next state depends on the last.

    .. rubric:: Notes

    **Pruning.** A prune drops an established summary lighter than
    ``beta_mu * w_bar``, and a summary not yet established lighter than
    DenStream's ``xi(age) * w_bar``. ``age`` is the clock that has passed
    since the summary opened, and ``h`` is the half-life:

    .. code-block:: text

        xi(age) = (2 ** (-(age + Tp) / h) - 1) / (2 ** (-Tp / h) - 1)
                = sum_{i <= age / Tp} 2 ** (-i Tp / h)        where age / Tp is whole
        Tp      = ceil(h log2(beta_mu / (beta_mu - 1)))

    That is the weight of a summary that had taken one row of the mean
    weight every ``Tp`` clock units since it opened. Both count rows, as
    DenStream's do: a stream with more rows to a clock unit fills its
    summaries faster. With no decay, or with ``beta_mu`` at most 1, where
    ``Tp`` has no value, no summary that is not yet established is pruned.
    An established summary is still dropped below ``beta_mu * w_bar``, as
    ``w_bar`` rises. Ids are monotone and never reused. A label is the
    smallest id in its chain, so it survives everything but the loss of that
    summary.

    **The prune schedule.** Without a clock column the clock is the
    row's number, the first row being 1, so ``prune_every=n`` prunes on
    the rows ``max_rows_between_prunes=n`` does while every row has weight.
    A row of weight zero advances the clock, so a quiet spell still prunes
    on time, and a gap capped at ``gap_cap`` counts as the cap. DenStream
    checks every ``Tp`` clock units (*Pruning*): give ``prune_every`` that
    to follow the paper.

    **The link.** Left out, ``macro_link`` derives ``L`` at each prune
    as 1.5 times the 90th percentile of the established summaries'
    nearest-neighbor
    distance, never below ``2 * eps * sqrt(p)``. So a chain along a shape holds
    without a constant that fragments one shape and bridges another.

    **The scale floor.** ``1 / var`` alone grows as ``2^Q`` over ``Q``
    half-lives of a feature going quiet (a flag that stops firing, a sensor
    at rest): a million at twenty, and the row on which the feature moves
    again is then infinitely far from every center. Floored, the weight
    grows as ``2^(Q/8) / scale_floor``, about 57 at twenty. The long-run
    variance is tracked at eight times the half-life, a feature at a time.
    Each row's weight and deviation are clipped against it, and its start
    is the medians of the feature's first five rows, so a row at the input
    bound moves it by a factor of 26 at most, which a few of its half-lives
    undo.
    """
    model: dict[str, Any] = {
        "type": "micro",
        "eps": eps,
        "beta_mu": beta_mu,
        "max_clusters": max_clusters,
        "prune_every": prune_every,
        "max_rows_between_prunes": max_rows_between_prunes,
        "macro_link": macro_link,
        "standardize": standardize,
        "scale_floor": scale_floor,
    }
    targets = _mirror_target(name, "micro", features, common, "its clusters are over")
    return _common(name, model, targets=targets, features=features, **common)
