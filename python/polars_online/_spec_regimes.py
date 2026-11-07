"""The regime and realised-covariance builders, :func:`hmm` and :func:`rcov`:
a part of :mod:`polars_online._spec` moved to a file of its own, keeping
that one under ``tests/test_repo_hygiene.py``'s 250 KB cap for a source
file. :mod:`polars_online.spec` documents them with the rest.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Literal, Unpack

from polars_online._kwargs import CommonKwargs
from polars_online._spec import _checked, _common, _mirror_target, _who
from polars_online._warnings import PolarsOnlineDeprecationWarning

#: The package's directory: the warning skips its frames, so it names the
#: caller's line.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


@_checked
def hmm(
    name: str,
    *,
    features: list[str],
    n_components: int,
    precision_prior: float,
    covariance_type: Literal["full", "tied", "diag"] = "full",
    learn: bool = True,
    transition_prior: float | None = None,
    transition: list[float] | None = None,
    means: list[float] | None = None,
    covs: list[float] | None = None,
    warm_rows: int | None = None,
    init: Literal["first", "farthest", "k-means++", "lloyd"] | None = None,
    seed: int | None = None,
    exog_tvtp: str | None = None,
    tvtp_coef: list[list[float]] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A Gaussian hidden Markov model of the features, filtered one row at a time.

    Use it to ask which regime the stream is in: a transition matrix carries
    what the rows before have shown into each row's answer. Use
    :func:`ew_class` instead to ask which labeled class a row looks like, when
    a column names each row's class.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`polars_online.spec`. This model reads ``features`` and takes no
    ``targets``. ``fit_intercept`` has nothing to act on, as there is no
    regression, and nothing that reads residuals applies. ``min_weight``
    withholds what is reported, never what is learned: a row below it moves
    the filter and shows nulls. This model's own:

    ``n_components``
        The number of hidden states, from 2 to 1,024; required.
    ``precision_prior``
        A ridge on every state's covariance, in the features' squared units,
        finite and above 0. It is required here as it is for
        :func:`ew_class`: a state's centered co-moments start at zero, a zero
        matrix has no density, and every row is predicted through each
        state's inverse. Keep it small against the data's scale: with states given
        and not learned, a ridge of 1.0 on data whose variance is about 1.0
        halves every correlation.
    ``covariance_type``
        ``"full"`` (the default), ``"tied"`` or ``"diag"``, as for
        :func:`ew_class`.
    ``learn``
        Whether the states and the transition counts move with the rows.
        Default ``True``. ``False`` with no states given is refused: there
        would be nothing to filter with.
    ``transition_prior``, ``transition``
        The Dirichlet pseudo-count per cell (default 1, finite and at least
        0), and an ``n_components x n_components`` matrix whose rows are
        distributions, to spread that mass over instead of flat, so the given
        matrix is the prior mean. A row with no counts is that mean, the
        given matrix or uniform, ``transition_prior = 0`` included. The prior
        pulls harder than its count suggests, and under
        ``transition_prior = 0`` a state the first rows do not favor can get
        no count (*The transition prior* in Notes). Both are refused beside
        ``tvtp_coef``, which learns no count.
    ``means``, ``covs``
        The states given outright: ``means`` is ``n_components x d`` and
        ``covs`` is ``n_components`` matrices of ``d x d`` (``d`` the number
        of features), both flattened row-major, finite, and given together.
        With them there is no warm-up. Each ``covs`` block must be symmetric
        and positive definite, and the pair enters at one row's weight. So
        under ``learn = True`` the stream washes the given states out as it
        would one row, within about one half-life, and under
        ``learn = False`` they are held exactly. Pass them where the regimes
        differ in their covariance alone (*A regime in the covariance alone*
        in Notes).
    ``warm_rows``, ``init``, ``seed``
        Without given states, the first ``warm_rows`` learned rows (default
        1,000, or ``n_components`` where that is more, and at least
        ``n_components``) are held to seed the states from.
        :func:`kmeans`' ``init`` (``"lloyd"`` by default; ``"first"``,
        ``"farthest"``, ``"k-means++"``) chooses centers among them with
        ``seed`` (default 0, at least 0; ``None`` is 0, so a run seeds the
        same states every time), and the held rows are replayed through those
        centers as hard assignments. A group with fewer rows than that never
        seeds, and reports null on every row. The held rows must hold every
        regime, several times over (*Seeding* in Notes). With ``means`` and
        ``covs`` given nothing is seeded, and the three are refused.
    ``exog_tvtp``, ``tvtp_coef``
        A column named on its own, as ``weight`` is, not listed in
        ``features``, whose value ``z_t`` drives the matrix instead: ``Pi_kl(t) =
        softmax_l(A_kl + B_kl z_t)`` from the fixed ``tvtp_coef = [A, B]``, each
        ``n_components x n_components`` flattened row-major. The two are given
        together. The count-based learning is off under it, and ``A`` and ``B`` are
        fitted elsewhere. A row whose ``exog_tvtp`` is null or not finite is
        filtered with ``z = 0``, the base transition, and is otherwise an ordinary
        row. :meth:`polars_online.ModelBank.predict` reads the column too.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields, all but ``coef`` read before the row is learned,
    so they are safe as features for that same row. `docs/OUTPUTS.md#hmm
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#hmm>`_
    gives each field's type and when it is null:

    ``filtered_state<j>``, ``predicted_state<j>``
        The filtered probability of each state the rows before left (uniform
        before the first), and the probability predicted for this row from
        it.
    ``state``
        The most likely state under the predicted probabilities.
    ``loglik``
        The row's surprise: its log-likelihood under the predicted mixture.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row, null where nothing decays.
    ``withheld_reason``
        Why the row's fields were withheld, null where they were not:
        ``below_warm_rows`` on the rows held to seed the states from, and
        ``below_min_weight`` on a row whose ``weight_sum`` is below
        ``min_weight``, which takes precedence where both hold.
    ``coef``
        The state means after the row is learned, one row per state, each
        one entry per feature (:func:`coef_index`). It reads zeros on the
        held rows before the last. The last held row reads the seeded means,
        since the states are seeded when it is learned.

    Until the states are seeded, every field but ``weight_sum``,
    ``settled_frac``, ``withheld_reason`` and ``coef`` is null. A row whose
    state densities cannot be evaluated is counted in
    :meth:`polars_online.ModelBank.solve_failures` and leaves the filter where
    it stands. Each seeding run that stopped at its iteration cap is counted
    there too.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), ``TypeError`` for
    ``targets``, and ``ValueError`` naming the problem for:

    - an ``n_components`` outside 2 to 1,024, a ``precision_prior`` not
      finite or not above 0, and a ``transition_prior`` below 0 or not
      finite;
    - a ``covariance_type`` other than ``"full"``, ``"tied"`` or ``"diag"``,
      and an ``init`` other than its four words;
    - ``learn = False`` without states;
    - ``means`` without ``covs`` or the reverse, either of the wrong shape or
      not finite, or a ``covs`` block that is not symmetric and positive
      definite;
    - a ``transition`` that is not ``n_components x n_components`` or has a
      row that is not a distribution;
    - a ``warm_rows`` below ``n_components``, and a ``seed`` below 0;
    - ``exog_tvtp`` without ``tvtp_coef`` or the reverse, and a
      ``tvtp_coef`` whose ``A`` and ``B`` are not
      ``n_components x n_components``;
    - ``transition`` or ``transition_prior`` beside ``tvtp_coef``, and
      ``warm_rows``, ``init`` or ``seed`` beside given states.

    .. rubric:: The update

    Hamilton's filter, one row at a time. Before the row, from the filtered
    ``p`` the previous row left (uniform before the first), with ``mu_l`` and
    ``Sigma_l`` state ``l``'s mean and covariance, ``omega_l`` its
    ``precision_prior`` ridge as it has decayed, ``Pi`` the transition
    matrix and ``x`` the row's features:

    .. code-block:: text

        p1_l   = sum_k p_k * Pi_kl                     the predicted state
        f_l    = N(x | mu_l, Sigma_l + omega_l I)      the state's density
        loglik = log sum_l p1_l f_l                    the row's surprise
        p_l   <- p1_l f_l / sum                        the filtered state

    Each state's running sums then take the row at weight ``w * p_l``, ``w`` the
    row's weight. The responsibilities sum to ``w``, so ``weight_sum`` follows
    the same recursion as in every model. The transition matrix is learned from
    the filtered joint of consecutive states. ``tau_kl`` is cell ``kl``'s
    Dirichlet pseudo-count: ``transition_prior``, or
    ``transition_prior * n_components * transition_kl`` with
    ``transition`` given. So a never-visited row of ``Pi`` is still a
    distribution, and a row with no counts and no pseudo-count
    (``transition_prior = 0``) reads ``transition``, or a uniform row
    without it.
    ``λ = 0.5 ** (Δclock / half_life)`` is the row's decay (``lam ** Δclock``
    under ``lam``). ``Δclock`` is the clock's step since the last row the
    model did not skip, capped at ``gap_cap``, or
    ``session_gap`` at a session change:

    .. code-block:: text

        N_kl <- λ * N_kl                               the row's decay, before the read
        Pi_kl = (N_kl + tau_kl) / sum_l (N_kl + tau_kl)   the matrix the row reads
        xi_kl = p_k(t-1) Pi_kl f_l / sum over all pairs
        N_kl <- N_kl + w * xi_kl

    .. rubric:: Example

    This code plants two regimes, at -2 and +2, between which a chain
    switches once in 50 rows on average, and prints the share of the rows
    after the warm-up whose ``state`` is their true regime:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(7)
        regime = np.zeros(5_000, dtype=int)
        for t in range(1, 5_000):                          # switches 1 row in 50
            regime[t] = regime[t - 1] if rng.random() > 0.02 else 1 - regime[t - 1]
        x = np.where(regime == 1, 2.0, -2.0) + rng.standard_normal(5_000)
        pl.DataFrame({"x": x, "regime": regime}).write_parquet("regimes.parquet")

        h = po.spec.hmm(
            "hmm", features=["x"], half_life=float("inf"),
            n_components=2,
            precision_prior=1e-2,                          # required: a state starts with no spread
            warm_rows=500,                                 # seeds the states from 500 rows
        )
        out = pl.scan_parquet("regimes.parquet").online.fit_predict([h]).collect()
        found = out["hmm"].struct.field("state")[500:].to_numpy()
        true = out["regime"][500:].to_numpy()
        share = max(np.mean(found == true), np.mean(found != true))   # the states may be swapped
        print(f"rows in their true regime: {share:.2f}")

    .. code-block:: text

        rows in their true regime: 0.97

    .. rubric:: See also

    - :func:`ew_class`, the same Gaussian densities against labeled classes.
    - :func:`kmeans`, whose ``init`` seeds the states.
    - `docs/REGIMES.md
      <https://github.com/hgilde/polars-online/blob/main/docs/REGIMES.md>`_,
      regime models measured on streams whose regimes are known.

    .. rubric:: Notes

    **Which regime are we in.** :func:`ew_class` classifies a row against
    labeled Gaussians. This does the same arithmetic with no labels: the
    state is hidden, and a transition matrix carries information from one
    row to the next. That is the difference between "which regime does this
    row look like" and "which regime are we in", and the second is usually
    the question. The densities go through the path :func:`ew_class` uses,
    with the same decaying ``precision_prior`` ridge.

    **The clock.** ``Pi`` applies once per row whatever the clock between
    rows, so a weekend between two rows moves the chain one step. The counts
    decay on the clock but grow by ``w`` per row, so the staying probability
    rises with the rows' density. A row reads ``Pi`` from the counts its own
    clock has decayed, so after a gap that takes them to nothing it reads the
    prior's mean. A row of weight 0 moves the clock and learns nothing, as
    in every model.

    **The transition prior.** The prior pulls harder than its count
    suggests. The counts are the filtered joint of consecutive states, read
    through the matrix being learned. A matrix pulled toward the prior mean
    is less sticky, so the filter credits more switches, and the counts
    follow it. The pull was measured on a sticky two-state chain, the states three
    noise standard deviations apart and learned from their true values. At
    ``transition_prior = 10``, 4,000 rows left the matrix 0.088 from the
    truth (the 95th percentile over 50 streams), where the pseudo-counts
    alone would move it about 0.006. The pull still fades with the rows, so
    the estimate is consistent. At the default the 95th percentile was 0.035,
    0.016 and 0.0066 at 4,000, 16,000 and 64,000 rows, about halving with
    each fourfold increase.

    One limit, accepted rather than refused: under ``transition_prior = 0``
    and ``learn = True`` the counts are only what the rows put there. A state
    the first rows do not favor can get no count, and the learned matrix then
    keeps it out. That needs no extreme separation: on the same chain, 7
    streams of 50 left a state under 10% of the counts, where the chain spends
    a third of its rows in it. None did at 0.1 or above. A prior above 0
    keeps every move possible.

    **Seeding.** The held rows age as ``weight_sum`` does, so the states
    start at the weight ``weight_sum`` says, not at the rows' raw weights. The
    seeding places the states where the held rows are, so a regime absent
    from them gets no state. Merging a regime with a few percent of the
    held rows into another regime's state adds less to the k-means inertia
    than splitting a larger regime removes, so k-means gives the larger
    regime the spare state and the small regime shares one. Learning does not move a
    misplaced state out. Measured on three regimes three noise standard
    deviations apart, each lasting 50 rows on average, over 50 seeds of 5,000
    rows: the worst share of rows in their true state was 0.95 from 1,000
    rows, 0.81 from 500 and 0.46 from 50. The model holds ``warm_rows`` rows
    of ``d`` values until the states are seeded, and the rows it holds report
    nothing.

    **Limits.** A single extreme row can be captured by one state, moving its
    mean far from the data. A state's running sums are weighted means, which
    a row of zero responsibility leaves as they are, so a state that stops
    winning never forgets, and the mixture is left short one state. To
    guard against it, increase ``precision_prior``, give the states with
    ``learn = False``, or clean the rows in the query before the bank.

    **A regime in the covariance alone.** A regime that lives only in the
    covariance needs the covariances to start from. The default seeding is k-means over
    the rows, and zero-mean states differ in nothing k-means can see, so it
    splits the rows by direction and the filter never recovers. On streams
    that stay in one of two zero-mean states, it puts 55% of the rows in the
    true state, the rows before seeding counted as right, and about half of
    the rows after seeding. The same filter given the true ``covs`` put 98%
    there, and 95% when it went on learning (`docs/REGIMES.md
    <https://github.com/hgilde/polars-online/blob/main/docs/REGIMES.md>`_
    §1). Pass ``means`` and ``covs``, or a feature in which the regime is a
    shift in location.
    """
    model: dict[str, Any] = {
        "type": "hmm",
        "n_components": n_components,
        "covariance_type": covariance_type,
        "precision_prior": precision_prior,
        "learn": learn,
        "transition_prior": transition_prior,
        "transition": transition,
        "means": means,
        "covs": covs,
        "warm_rows": warm_rows,
        "init": init,
        "seed": seed,
        "exog_tvtp": exog_tvtp,
        "tvtp_coef": tvtp_coef,
    }
    mirror = _mirror_target(name, "hmm", features, common, "its hidden states are over")
    targets = [exog_tvtp] if exog_tvtp is not None else mirror
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def rcov(
    name: str,
    *,
    features: list[str],
    kind: Literal["kernel", "preavg", "plain"] = "kernel",
    kernel: Literal["parzen"] = "parzen",
    bandwidth: int | None = None,
    jitter: int | None = None,
    theta: float | None = None,
    psd: bool = True,
    block_rows: int | None = None,
    max_bandwidth: int | None = None,
    preavg_rows: int | None = None,
    noise_stride: int | None = None,
    iv_stride: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A block's realized covariance of the features, robust to microstructure noise.

    Use it for the covariance of returns over each block of a tick stream,
    such as a trading day, where the noise in each price biases a plain
    realized covariance. Use :func:`ew_cov` for running moments that decay
    and are read on every row.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``gap_cap`` and ``min_weight``, are in :mod:`polars_online.spec`. This
    model reads ``features``, whose rows are returns: compute them with
    Polars' ``diff()`` in the query before the bank (``.diff().over(by)``
    after :func:`polars_online.stream.refresh_time`). It takes no
    ``targets``, and ``fit_intercept`` has nothing to act on, as there is no
    regression. It requires ``group`` and ``group_close``, since the
    estimate rides in the row the close emits. It refuses ``half_life`` and
    ``lam``, since ``group_close`` ends each block and no decay applies.
    ``weight`` is taken only as 0 or 1, since a sum over returns has no
    fractional row, and a zero-weight row advances the clock and enters no
    ring. This model's own:

    ``kind``, ``kernel``
        The estimator: ``"plain"``, ``"kernel"`` (the default) or
        ``"preavg"``, as *The update* gives them. ``kernel`` takes only
        ``"parzen"``.
    ``bandwidth``, ``block_rows``, ``max_bandwidth``
        ``bandwidth`` is a fixed ``H``. Left out, it is chosen from the
        block, as *Bandwidth* in Notes says, and it needs ``block_rows``: the
        ring has to be sized before the first row, and a block's length is
        known only at the close. ``block_rows`` is a sizing hint, not a
        limit: a longer block runs, its bandwidth clipped to the ring, and
        reports
        ``rcov_bandwidth_used``. ``max_bandwidth`` fixes the ring depth
        itself (default ``ceil(c* block_rows^(3/5))`` under the automatic
        bandwidth, the depth at which the noise equals the block's integrated
        variance), and must not cap the ring below a fixed ``bandwidth``. The
        ring's lagged products, ``(ring + 1) * k^2`` doubles for ``k``
        features, are held to 256 MiB before the first row. ``bandwidth`` and
        ``max_bandwidth`` apply under ``"kernel"`` alone, and are refused
        under the other kinds. ``block_rows`` also sizes ``"preavg"``'s window.
    ``jitter``
        Observations averaged at each end. Default 2, and ``1`` is no
        jitter. ``jitter`` values from 1 to 4, which Barndorff-Nielsen,
        Hansen, Lunde and Shephard try, move the estimate by under 0.5%.
        ``"kernel"`` only: the other kinds read no ends, and refuse it.
    ``theta``, ``psd``, ``preavg_rows``
        ``"preavg"``'s window scale (default 1.0, finite and above 0), form
        and override. ``theta`` and ``preavg_rows`` are refused under the
        other kinds, which read no window. Under ``"preavg"``,
        ``psd = False`` is the balanced, bias-corrected form (optimal rate,
        not guaranteed positive semi-definite), and ``psd = True`` (the
        default) the longer window ``k_n = ceil(theta * block_rows^0.6)``
        without the bias term. Under every kind, ``psd = True`` clips any
        negative eigenvalue of the estimate, reporting
        ``rcov_psd_repaired``. ``preavg_rows`` fixes ``k_n`` instead of
        deriving it from ``block_rows``: at least 2, and at least 3 under
        ``psd = False``, where a window of 2 leaves nothing once the bias is
        subtracted. A window ``theta`` derives is at least 2, at least 3
        under ``psd = False``, and no longer than ``block_rows``. Beside
        ``preavg_rows``, which fixes the window, ``theta`` has no effect: it
        is deprecated there, issuing a
        :class:`~polars_online.PolarsOnlineDeprecationWarning`, and is
        accepted and ignored until the next major version, which refuses it.
    ``noise_stride``, ``iv_stride``
        The strides of two subsampled grids, defaults 1 and 20, each at
        least 1. The dense one gives the noise variance ``rcov_omega2``,
        deliberately biased upward as Barndorff-Nielsen, Hansen, Lunde and
        Shephard accept, and the sparse one the integrated variance
        ``rcov_iv_sparse``, each averaged over the stride's offsets. Both
        are reported under every kind, and an automatic bandwidth reads
        them.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record holding ``weight_sum`` alone. `docs/OUTPUTS.md#rcov
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#rcov>`_
    gives its type. The estimate is in the row
    :meth:`polars_online.ModelBank.closed_groups` returns when the group
    closes:

    ``rcov``, ``rcorr``
        The covariance and correlation estimates, as ``vech`` of the upper
        triangle.
    ``rcov_n``, ``rcov_kind``, ``rcov_bandwidth_used``
        The effective returns, the estimator, and the ``H`` used. The
        effective returns are the returns under ``"plain"``; the jittered
        returns under ``"kernel"``, ``n - 2 * jitter + 2`` of a block of
        ``n`` with no break; and the pre-averaged terms under ``"preavg"``,
        ``n - k_n + 2`` per stretch.
    ``rcov_omega2``, ``rcov_iv_sparse``
        The noise variance and the sparse integrated variance, per feature,
        from the grids ``noise_stride`` and ``iv_stride`` set; an automatic
        bandwidth reads them.
    ``rcov_iq``
        Per feature, the realized quarticity ``(n / 3) * sum(x**4)`` on the
        sparse grid of ``iv_stride``, averaged over its offsets, with ``n``
        the grid's returns: a proxy for the integrated quarticity, and named
        as one.
    ``rcov_psd_repaired``
        Whether a negative eigenvalue was clipped. Null where the repair
        could not run, an entry that is not finite or an eigendecomposition
        that failed. The matrix is then reported as it stands.

    For a block too short to estimate from, ``rcov`` and ``rcorr`` are null,
    and the other fields say what the block held. ``min_weight`` (default 0)
    withholds them the same way from a block with fewer returns than
    ``min_weight``. A row's
    weight is 0 or 1, so the block's ``weight_sum`` is its count of returns:
    a block of 40 returns under ``min_weight=50`` reports null ``rcov``
    and ``rcorr`` beside its ``rcov_n`` and the noise fields.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), ``TypeError`` for
    ``targets``, and ``ValueError`` naming the problem for:

    - ``half_life`` or ``lam``, and a missing ``group`` or ``group_close``;
    - a ``kind`` other than its three words, and a ``kernel`` other than
      ``"parzen"``;
    - an automatic bandwidth without ``block_rows``, and ``"preavg"`` with
      neither ``block_rows`` nor ``preavg_rows``;
    - a ``block_rows``, ``jitter`` or stride of 0, and a ``theta`` that is
      not finite or not above 0;
    - a ``max_bandwidth`` below a fixed ``bandwidth``, and a
      ``preavg_rows`` below 2, or below 3 under ``psd = False``;
    - a ``bandwidth``, ``max_bandwidth`` or ``preavg_rows`` above
      ``block_rows``, and a ``theta`` whose window is longer than
      ``block_rows``;
    - a ring, window, ``jitter`` or stride above 2^20 (1,048,576), the most
      this model sizes a ring, and lagged products past 256 MiB;
    - ``bandwidth``, ``max_bandwidth`` or ``jitter`` under a kind other
      than ``"kernel"``, and ``theta`` or ``preavg_rows`` under one other
      than ``"preavg"``.

    And, from a call that learns, ``ValueError`` for a ``weight`` other than
    0 or 1, naming the row.

    .. rubric:: The update

    Each row's returns ``x`` enter the block's sums over lags, and at the
    close the estimator reads them. ``"plain"`` is ``sum(x x')``, which at
    close equals ``n`` times the uncentered second moment of an
    ``ew_cov(lam=1)``, to the bit: the cross-check, and the reference the
    other two are measured against.

    ``"kernel"`` (the default) is Barndorff-Nielsen, Hansen, Lunde and
    Shephard's multivariate realized kernel, with the Parzen weight ``k``
    and bandwidth ``H``:

    .. code-block:: text

        K = sum_{h=-H}^{H} k(h / (H + 1)) * Gamma_h
        Gamma_h = sum_j x_j x'_{j-h},  Gamma_{-h} = Gamma_h'

    The end points are jittered by averaging the first and last ``jitter``
    observations.

    ``"preavg"`` is Christensen, Kinnebrock and Podolskij's modulated
    realized covariance: the returns are pre-averaged over ``k_n`` rows with
    ``g(x) = min(x, 1 - x)``, which averages the noise away. By default
    (``psd = True``) the window is the longer
    ``k_n = ceil(theta * block_rows^0.6)`` and no bias term is subtracted,
    their positive semi-definite form. ``psd = False`` is the balanced
    ``k_n = floor(theta * sqrt(block_rows))`` with the residual bias
    subtracted, and rescaled as their footnote 1 says. Over the block's ``n``
    returns and its ``rcov_n`` pre-averaged terms, with their finite-sample
    ``psi1`` and ``psi2`` (1 and 1/12 in the limit):

    .. code-block:: text

        Ybar_i = sum_{j=1}^{k_n - 1} g(j / k_n) x_{i+j}
        MRC = n / rcov_n / (psi2 k_n) * sum_i Ybar_i Ybar_i'
        b = psi1 / (2 psi2 k_n^2)
        psd = True:   MRC
        psd = False:  (MRC - b * sum_j x_j x_j') / (1 - b)

    .. rubric:: Example

    This code plants two prices whose returns have a variance of 1e-4 a tick
    and a correlation of 0.5, and adds noise to each price. Over the first
    block of 5,000 ticks, whose true variance is 0.5, it prints what the
    default kernel and the plain sum find:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(8)
        n = 20_000                                            # 4 blocks of 5,000 ticks
        a, b = rng.standard_normal(n), rng.standard_normal(n)
        p0 = np.cumsum(0.01 * a)                              # two prices, correlated 0.5
        p1 = np.cumsum(0.01 * (0.5 * a + np.sqrt(0.75) * b))
        noise = 0.01 * rng.standard_normal((2, n))            # noise on each price
        block = np.arange(n) // 5_000
        prices = pl.DataFrame({"p0": p0 + noise[0], "p1": p1 + noise[1], "block": block})
        prices.write_parquet("ticks_noisy.parquet")

        returns = pl.scan_parquet("ticks_noisy.parquet").with_columns(
            pl.col("p0", "p1").diff().over("block")          # rows are returns
        ).drop_nulls()
        common = dict(
            features=["p0", "p1"], group="block", group_close="monotone", block_rows=5_000,
        )
        bank = po.ModelBank([
            po.spec.rcov("rk", **common),                     # the kernel, the default
            po.spec.rcov("plain", kind="plain", **common),    # the plain sum, for comparison
        ])
        bank.fit(returns)                                     # each estimate comes at its close
        print("the first block's truth: variance 0.50, correlation 0.50")
        for r in bank.closed_groups().filter(pl.col("group") == "0").iter_rows(named=True):
            print(f"{r['spec']}: variance {r['rcov'][0]:.2f}, correlation {r['rcorr'][1]:.2f}")

    .. code-block:: text

        the first block's truth: variance 0.50, correlation 0.50
        rk: variance 0.51, correlation 0.46
        plain: variance 1.50, correlation 0.17

    .. rubric:: See also

    - :func:`ew_cov`, running moments that decay, read before each row.
    - :func:`polars_online.stream.refresh_time`, to put the prices of
      several series on one grid before taking returns.
    - :meth:`polars_online.ModelBank.closed_groups`, which returns each
      block's estimate.

    .. rubric:: Notes

    **Why it is not the plain sum.** A plain realized covariance over ticks
    is biased by noise (each price is the efficient one plus an error, and
    the error's variance accumulates with every tick) and attenuated by
    asynchrony. Both are estimated away by published estimators that are
    sums over lags, which is exactly what a stream can accumulate. There is
    no decay and no output on a row but ``weight_sum``, because the estimate
    describes a whole block and exists only once the block closes.

    **The kernel.** The Parzen kernel is a positive-definite function, so
    ``K`` is positive semi-definite up to rounding, and 0.97 efficient
    against the quadratic spectral's 0.93. The Bartlett kernel is not
    consistent here and is not offered.

    **Bandwidth.** Left out, ``bandwidth`` is their
    ``H = ceil(c* xi^(4/5) n^(3/5))`` with ``c* = 3.5134``, computed per
    feature with ``xi^2 = omega2 / iv_sparse`` and averaged over the
    features, and 1 where no feature has one.

    **Pre-averaging.** ``theta`` sets the window and nothing else: the bias
    term and its rescaling read theta from the window actually run,
    ``k_n / sqrt(n)`` over the block's own ``n`` rows (the paper's Eq. 7).
    ``sum(x x')`` holds the block's integrated covariance as well as the
    noise, so subtracting ``b`` of it removes ``b`` of the covariance too.
    Dividing by ``1 - b`` restores it, where the undivided estimate averages
    0.84 of the covariance at ``k_n = 6``.

    **Breaks in a block.** A gap over ``gap_cap``, or a session change,
    splits the block into stretches: the returns on either side of it are
    not adjacent, and a covariance of adjacent returns is the whole
    statistic. Each stretch is closed as the last one is (leading jitter,
    interior, trailing jitter), and the lagged sums add over stretches, so no
    product pairs two returns across the break. The two subsampled grids
    start again at the break as well, so no step of ``omega2``,
    ``iv_sparse`` or ``iq`` sums returns across it either. Under
    ``session_gap = "reset"`` a session change starts the block over
    instead, and drops the returns before it. A stretch too short to reach
    its trailing jitter contributes only what it had already emitted, and
    ``rcov_n`` says how many effective returns there were in total. Nothing
    reads a future row: the jittered end point is formed at close from
    observations already in the state.
    """
    model: dict[str, Any] = {
        "type": "rcov",
        "kind": kind,
        "kernel": kernel,
        "bandwidth": bandwidth,
        "jitter": jitter,
        "theta": theta,
        "psd": psd,
        "block_rows": block_rows,
        "max_bandwidth": max_bandwidth,
        "preavg_rows": preavg_rows,
        "noise_stride": noise_stride,
        "iv_stride": iv_stride,
    }
    targets = _mirror_target(name, "rcov", features, common, "its covariance is of")
    out = _common(name, model, targets=targets, features=features, **common)
    if theta is not None and preavg_rows is not None:
        # Inert since the window is all `theta` sets: deprecated rather than
        # refused, as the user decided (task 278, b11).
        warnings.warn(
            f"{_who(name)}: theta={theta!r} is ignored beside preavg_rows, which fixes the "
            "pre-averaging span theta would set. The next major version refuses it. "
            "Leave theta out.",
            PolarsOnlineDeprecationWarning,
            skip_file_prefixes=(_PACKAGE_DIR,),
        )
    return out
