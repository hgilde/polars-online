"""The regime and realised-covariance builders, :func:`hmm` and :func:`rcov`:
a part of :mod:`polars_online._spec` moved to a file of its own, keeping
that one under ``tests/test_repo_hygiene.py``'s 250 KB cap for a source
file. :mod:`polars_online.spec` documents them with the rest.
"""

from __future__ import annotations

from typing import Any, Unpack

from polars_online._kwargs import CommonKwargs
from polars_online._spec import _checked, _common, _mirror_target


@_checked
def hmm(
    name: str,
    *,
    features: list[str],
    k: int,
    precision_prior: float,
    covariance: str = "full",
    learn: bool = True,
    transition_prior: float | None = None,
    transition: list[float] | None = None,
    means: list[float] | None = None,
    covs: list[float] | None = None,
    warm_rows: int | None = None,
    seed_rule: str | None = None,
    seed: int | None = None,
    exog_tvtp: str | None = None,
    tvtp_coef: list[list[float]] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A Gaussian hidden Markov model, filtered online: which regime are we in?

    :func:`ew_class` classifies a row against labelled Gaussians. This does
    the same arithmetic with no labels: the state is hidden, and a
    transition matrix carries information from one row to the next. That is
    the difference between "which regime does this row look like" and
    "which regime are we in", and the second is usually the question. Not a
    regression: no targets, and nothing residual-based applies.

    .. rubric:: The recursion

    Hamilton's filter, one row at a time. Before the row, from the filtered
    ``p`` the previous row left (uniform before the first):

    .. code-block:: text

        p1_l   = sum_k p_k * Pi_kl                     the predicted state
        f_l    = N(x | mu_l, Sigma_l + r_l I)          the state's density
        loglik = log sum_l p1_l f_l                    the row's surprise
        p_l   <- p1_l f_l / sum                        the filtered state

    Everything reported is read before the row is learned from, so an
    ``hmm`` output is safe as a feature for that same row. The densities go
    through the path :func:`ew_class` uses, with the same decaying
    ``precision_prior`` ridge. Each state's accumulator then takes the row
    at weight ``w * p_l``; the responsibilities sum to ``w``, so
    ``weight_sum`` is the shared recursion untouched. The transition matrix
    is learned from the filtered joint of consecutive states:

    .. code-block:: text

        A_kl <- decay * A_kl                           the row's decay, before the read
        Pi_kl = (A_kl + tau_kl) / sum_l (A_kl + tau_kl)   the matrix the row reads
        xi_kl = p_k(t-1) Pi_kl f_l / sum over all pairs
        A_kl <- A_kl + w * xi_kl

    with ``tau`` a Dirichlet pseudo-count per cell, which is what keeps a
    never-visited row of ``Pi`` a distribution. A transition is one row:
    ``Pi`` applies once per row whatever the clock between rows, so a
    weekend is one step. The counts decay on the clock but grow by ``w`` per
    row, so the staying probability rises with the rows' density. A row
    reads ``Pi`` from the counts its own clock has decayed, so after a gap
    that takes them to nothing it reads the prior's mean, and a zero-weight
    row is clock alone, as in every model.

    Two limitations worth knowing. A single extreme row can be captured by
    one state, moving its mean far from the data. In mean form a state with
    zero responsibility keeps its moments, so a state that stops winning
    never forgets, and the mixture is left short one state. A larger
    ``precision_prior``, given states (``learn = False``) or cleaning
    upstream are the mitigations. And a regime that lives only in the
    covariance needs the covariances to start from. The default seeding is
    k-means over the rows, and zero-mean states differ in nothing k-means
    can see, so it splits the rows by direction and the filter never
    recovers. On streams that stay in one of two zero-mean states, it puts
    57% of the rows in the true state, about half of those after seeding,
    against 98% for the same filter given ``covs`` (`docs/REGIMES.md
    <https://github.com/hgilde/polars-online/blob/main/docs/REGIMES.md>`_
    §1). Pass ``means`` and ``covs``, or a feature in which the regime is a
    shift in location. And keep ``precision_prior`` small against the data's
    scale: with states given and not learned, a ridge of 1.0 on data whose
    variance is about 1.0 halves every correlation.

    .. rubric:: Parameters

    ``k``
        The number of hidden states; required.
    ``precision_prior``
        A ridge on every state's covariance, required here as it is for
        :func:`ew_class`, because a state's centred co-moments start at zero
        and a zero matrix has no density.
    ``covariance``
        ``"full"`` (the default), ``"shared"`` or ``"diagonal"``, as for
        :func:`ew_class`.
    ``learn``
        Whether the states move with the rows. ``False`` with no states given
        is refused: there would be nothing to filter with.
    ``transition_prior``, ``transition``
        The Dirichlet pseudo-count per cell (default 1), and a matrix to
        spread that mass over instead of flat, so the given matrix is the
        prior mean. A row with no counts is that mean, the given matrix or
        uniform, ``transition_prior = 0`` included. Both are refused beside
        ``tvtp_coef``, which learns no count.
    ``means``, ``covs``
        The states given outright (``K x d`` and ``K`` matrices of ``d x d``,
        both flattened row-major), and then there is no warm-up. Each
        ``covs`` block must be symmetric and positive definite, and the pair
        enters at one row's weight. So under ``learn = True`` the stream
        washes the given states out at the ordinary rate, and under ``learn
        = False`` they are held exactly.
    ``warm_rows``, ``seed_rule``, ``seed``
        Without given states, the first ``warm_rows`` learned rows (default
        50) are buffered. :func:`kmeans`' ``seed_rule`` (``"lloyd"`` by
        default; ``"first"``, ``"farthest"``, ``"kmeanspp"``) chooses centres
        among them with ``seed`` (default 0), and the buffer is replayed
        through those centres as hard assignments. Every output is null
        until then. The buffered rows age as ``weight_sum`` does, so the
        states start at the weight ``weight_sum`` says, not at the rows' raw
        weights. The buffer should span more than one regime, or the seeds
        are two halves of one. With ``means`` and ``covs`` given nothing is
        seeded, and the three are refused.
    ``exog_tvtp``, ``tvtp_coef``
        A column (declared like ``weight``, not a feature) whose value drives
        the matrix instead: ``Pi_kl(t) = softmax_l(A_kl + B_kl z_t)`` from
        the fixed ``tvtp_coef = [A, B]``. The count-based learning is off
        under it; ``A`` and ``B`` are fitted elsewhere. A row whose
        ``exog_tvtp`` is null or non-finite is filtered with ``z = 0``, the
        base transition, and is otherwise an ordinary row.
        :meth:`polars_online.ModelBank.predict` reads the column too.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest.

    ``min_weight`` gates what is reported, never what is learned: a row
    below it moves the filter and shows nulls. A row whose state densities
    cannot be evaluated is counted in
    :meth:`polars_online.ModelBank.solve_failures` and leaves the filter
    where it stands.

    .. rubric:: Output

    One struct column named after the spec, all read before the row is
    learned (`docs/OUTPUTS.md#hmm
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#hmm>`_):

    ``filtered_<j>``, ``predicted_<j>``
        The filtered and the predicted probability of each state.
    ``state``
        The most likely state.
    ``loglik``
        The row's surprise: its log-likelihood under the predicted mixture.
    ``weight_sum``
        As everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where
        nothing decays, and why the row's fields are null where they are, as
        everywhere.
    ``coef``
        The state means, one row per state, each one entry per feature
        (:func:`coef_index`).

    .. rubric:: Example

    .. code-block:: python

        h = po.spec.hmm(
            "regime", features=["x0", "x1"], half_life=500.0,
            k=2,
            precision_prior=1e-2,    # required: a zero matrix has no density
            warm_rows=100,           # seeds the states from this many rows; null until then
        )
        out = po.ModelBank([h]).fit_predict(df).unnest("regime")
        states = out.select("filtered_0", "filtered_1", "state", "loglik").tail(3)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``; ``ValueError`` for:

    - ``learn = False`` without states;
    - ``means`` or ``covs`` of the wrong shape, or a ``covs`` block that is
      not positive definite;
    - ``transition`` that is not ``k x k``;
    - ``transition`` or ``transition_prior`` beside ``tvtp_coef``, and
      ``warm_rows``, ``seed_rule`` or ``seed`` beside given states.
    """
    model: dict[str, Any] = {
        "type": "hmm",
        "k": k,
        "covariance": covariance,
        "precision_prior": precision_prior,
        "learn": learn,
        "transition_prior": transition_prior,
        "transition": transition,
        "means": means,
        "covs": covs,
        "warm_rows": warm_rows,
        "seed_rule": seed_rule,
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
    kind: str = "kernel",
    kernel: str = "parzen",
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
    """A block's realised covariance, robust to microstructure noise.

    A plain realised covariance over ticks is biased by noise (each price is the
    efficient one plus an error, and the error's variance accumulates with every
    tick) and attenuated by asynchrony. Both are estimated away by published
    estimators that are sums over lags, which is exactly what a stream can
    accumulate. Rows are returns: difference upstream (``.diff().over(by)`` after
    :func:`polars_online.stream.refresh_time`). There is no decay and no per-row
    output but ``weight_sum``, because the value is the block: ``rcov`` requires
    ``group`` and ``group_close``, and the estimate rides in the row that close
    emits.

    .. rubric:: The estimators

    ``"plain"`` is ``sum(x x')``, which at close equals ``n`` times an
    ``ew_cov(lam=1)``'s uncentred second moment to the bit -- the cross-check, and
    the reference the other two are measured against.

    ``"kernel"`` (the default) is Barndorff-Nielsen, Hansen, Lunde & Shephard's
    multivariate realised kernel:

    .. code-block:: text

        K = sum_{h=-H}^{H} k(h / (H + 1)) * Gamma_h
        Gamma_h = sum_j x_j x'_{j-h},  Gamma_{-h} = Gamma_h'

    with the Parzen kernel, a positive-definite function, so ``K`` is PSD up to
    rounding, and 0.97 efficient against the quadratic spectral's 0.93. The
    Bartlett kernel is not consistent here and is not offered. The end points are
    jittered by averaging the first and last ``jitter`` observations.

    ``"preavg"`` is Christensen, Kinnebrock & Podolskij's modulated realised
    covariance: the returns are pre-averaged over ``k_n`` rows with ``g(x) =
    min(x, 1 - x)``, which averages the noise away. By default (``psd =
    True``) the window is the longer ``k_n = ceil(theta * block_rows^0.6)``
    and no bias term is subtracted, their positive semi-definite form.
    ``psd = False`` is the balanced ``k_n = floor(theta * sqrt(block_rows))``
    with the residual bias subtracted, and rescaled as their footnote 1 says:

    .. code-block:: text

        Ybar_i = sum_{j=1}^{k_n - 1} g(j / k_n) x_{i+j}
        MRC = n / rcov_n / (psi2 k_n) * sum_i Ybar_i Ybar_i'
        b = psi1 / (2 psi2 k_n^2)
        psd = True:   MRC
        psd = False:  (MRC - b * sum_j x_j x_j') / (1 - b)

    over the block's ``n`` returns and its ``rcov_n`` pre-averaged terms, with
    their finite-sample ``psi1`` and ``psi2`` (1 and 1/12 in the limit).
    ``sum(x x')`` holds the block's integrated covariance as well as the
    noise, so subtracting ``b`` of it removes ``b`` of the covariance too.
    Dividing by ``1 - b`` restores it, where the undivided estimate averages
    0.84 of the covariance at ``k_n = 6``.

    .. rubric:: Parameters

    ``kind``, ``kernel``
        ``"plain"``, ``"kernel"`` (the default) or ``"preavg"``; ``kernel`` takes
        only ``"parzen"``.
    ``bandwidth``, ``block_rows``, ``max_bandwidth``
        ``bandwidth`` is a fixed ``H``. Left out, it is their ``H = ceil(c*
        xi^(4/5) n^(3/5))`` with ``c* = 3.5134``, which needs ``block_rows``: the
        ring has to be sized before the first row and ``n`` is known only at the
        close. ``block_rows`` is a sizing hint, not a limit: a longer block runs,
        clipped, and reports ``rcov_bandwidth_used``. ``max_bandwidth`` fixes the ring
        depth itself (default ``ceil(c* block_rows^(3/5))`` under the automatic
        bandwidth, the depth at which the noise equals the block's integrated
        variance); it must not cap the ring below a fixed ``bandwidth``. The
        ring's lagged products, ``(ring + 1) * k^2`` doubles for ``k``
        features, are held to 256 MiB before the first row. All three are
        ``"kernel"``'s, and refused under the other kinds.
    ``jitter``
        Observations averaged at each end. Default 2; ``1`` is no jitter, and the
        paper's own ``m = 1..4`` move the estimate by under 0.5%. ``"kernel"``
        only: the other kinds read no ends, and refuse it.
    ``theta``, ``psd``, ``preavg_rows``
        ``"preavg"``'s window scale (default 1.0; refused under the other
        kinds, which read no window), form and override. ``psd =
        False`` is the balanced, bias-corrected form (optimal rate, not guaranteed
        PSD); ``psd = True`` (the default) is the longer window ``k_n =
        ceil(theta * block_rows^0.6)`` without the bias term, and clips any
        negative eigenvalue, reporting ``rcov_psd_repaired``. ``preavg_rows`` fixes
        ``k_n`` instead of deriving it from ``block_rows``: at least 2, and at
        least 3 under ``psd = False``, where a window of 2 leaves nothing once
        the bias is subtracted.
        ``theta`` sets the window and nothing else: the bias term and its
        rescaling read theta from the window actually run, ``k_n / sqrt(n)``
        over the block's own ``n`` rows (the paper's Eq. 7). A window
        ``theta`` derives is at least 2, at least 3 under ``psd = False``, and
        no longer than ``block_rows``.
    ``noise_stride``, ``iv_stride``
        The two subsampled grids behind an automatic bandwidth (defaults 1 and
        20): the noise variance ``omega2`` from the dense one, deliberately biased
        upward as BNHLS accept, and the integrated variance ``iv_sparse`` from the
        sparse one, each averaged over the stride's offsets.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest.

    ``weight`` is taken only as 0 or 1, since a sum over returns has no fractional
    row, and a zero-weight row advances the clock and enters no ring.
    ``half_life``/``lam`` are refused: the block boundary is ``group_close``'s, not
    a decay's. A gap over ``gap_cap``, or a session change, splits the block
    into stretches: the returns on either side of it are not adjacent, and a
    covariance of adjacent returns is the whole statistic. Each stretch is closed
    as the last one is (leading jitter, interior, trailing jitter), and the lagged
    sums add over stretches, so no product pairs two returns across the break.
    The two subsampled grids start again at the break as well, so no step of
    ``omega2``, ``iv_sparse`` or ``iq`` sums returns across it either. A
    stretch too short to reach its trailing jitter contributes only what it had
    already emitted, and ``rcov_n`` says how many effective returns there were in
    total. Nothing reads a future row: the jittered end point is formed at close
    from observations already in state.

    .. rubric:: Output

    One struct column named after the spec holding ``weight_sum`` alone
    (`docs/OUTPUTS.md#rcov
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#rcov>`_).
    The estimate is in the row :meth:`polars_online.ModelBank.closed_groups` emits
    when the group closes:

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
        The noise variance and the sparse integrated variance behind an automatic
        bandwidth.
    ``rcov_iq``
        A realised-quarticity proxy, and labelled one.
    ``rcov_psd_repaired``
        Whether a negative eigenvalue was clipped. Null where the repair
        could not run, an entry that is not finite or an eigendecomposition
        that failed; the matrix is then reported as it stands.

    All null for a block too short to estimate from.

    .. rubric:: Example

    .. code-block:: python

        by_block = df.with_columns(block=pl.int_range(pl.len()) // 100)   # four blocks
        r = po.spec.rcov(
            "rk", features=["x0", "x1"],           # rows are returns
            group="block", group_close="monotone",
            kind="kernel",       # the multivariate realised kernel with Parzen weights
            block_rows=100,      # a sizing hint for the ring; a longer block runs, clipped
        )
        bank = po.ModelBank([r])
        bank.fit_predict(by_block.select("x0", "x1", "block"))
        blocks = bank.closed_groups()    # rcov, rcorr, rcov_n, rcov_bandwidth_used, ...


    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, ``ValueError`` for ``half_life``/``lam``, for a missing ``group``
    or ``group_close``, for an automatic bandwidth without ``block_rows``, for
    a ``max_bandwidth`` below a fixed ``bandwidth``, for a ``preavg_rows``
    below 2, or below 3 under ``psd = False``, for a ``bandwidth``,
    ``max_bandwidth`` or ``preavg_rows`` above ``block_rows``, for a ``theta``
    whose window is longer than ``block_rows``, for a ring, window,
    ``jitter`` or stride above 2^20 (1,048,576), the most this model sizes a
    ring, for lagged products past 256 MiB, and for ``jitter``,
    ``max_bandwidth`` or ``theta`` under a kind that does not read it.
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
    return _common(name, model, targets=targets, features=features, **common)
