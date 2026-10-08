"""E59: has the correlation structure changed?

Three tests with three nulls. The `monitor` kind is Wied, Krämer & Dehling's
closed-sample constancy test, and its critical value is a *published*
distribution — so its size and power are held to their tables rather than to
a number this implementation happened to produce. The `sequential` kind is
Wied & Galeano's detector against a stable history (docs/PLAN.md task 114),
whose critical values are held to their Table 1 and to a simulation of the
law behind it. The `window` kind measures how big a change is, against a
permutation null; that one is held to a longhand statistic and to behaving
on a stationary stream.
"""

import itertools

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"


def pair(n, rho, seed=0, k=2):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 1))
    x = np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.standard_normal((n, k))
    return pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})


def spec(**kw):
    d = {"features": ["x0", "x1"], "span_rows": 200}
    d.update(kw)
    return po.spec.corrchange("c", **d)


def run(df, **kw):
    return po.ModelBank([spec(**kw)]).fit_predict(df)["c"].struct.unnest()


# --- the monitor -------------------------------------------------------------


def test_the_critical_value_is_the_kolmogorov_quantile():
    """Computed from the series, not pinned: 1.3581 at 5%, 1.6276 at 1%,
    1.2239 at 10% are the published values it must reproduce."""
    for alpha, want in ((0.05, 1.3581), (0.01, 1.6276), (0.10, 1.2239)):
        out = run(pair(200, 0.4), span_rows=200, alpha=alpha, alpha_adjust="none")
        assert out["crit"][199] == pytest.approx(want, abs=1e-3)
    # Bonferroni over the pairs takes a smaller level, so a larger value.
    three = run(
        pair(200, 0.4, k=3),
        features=["x0", "x1", "x2"],
        span_rows=200,
        alpha=0.05,
    )
    assert three["crit"][199] > 1.3581


def test_nothing_is_reported_except_where_a_span_closes():
    out = run(pair(250, 0.3), span_rows=100)
    live = [i for i, v in enumerate(out["stat"].to_list()) if v is not None]
    assert live == [99, 199], live
    assert out["flag"][99] in (True, False)
    assert out["flag"][50] is None


def test_a_constant_correlation_is_not_flagged_and_a_break_is():
    """The whole point, at a break the test is built to find."""
    n = 500
    steady = run(pair(n, 0.5, seed=1), span_rows=n, alpha_adjust="none")
    assert steady["flag"][n - 1] is False, steady["stat"][n - 1]
    a = pair(n // 2, 0.2, seed=2)
    b = pair(n // 2, 0.9, seed=3)
    broken = run(pl.concat([a, b]), span_rows=n, alpha_adjust="none")
    assert broken["flag"][n - 1] is True
    assert broken["stat"][n - 1] > broken["crit"][n - 1]


def test_the_size_is_near_nominal_on_gaussian_pairs():
    """On **Gaussian** pairs the rejection rate is near the 5% it asks for.

    That is the property to hold here, and it is not quite the paper's
    table. WKD's Table 1 is for "i.i.d. bivariate `t_5` innovations" and
    reads `.040 / .035 / .041` at `T = 500` for `rho = -0.5 / 0 / 0.5`. The
    phrase does not pin the distribution down, but both readings of it, a
    shared-scale multivariate `t_5` and independent `t_5` marginals, land
    within 0.011 of that table, and neither is liberal. `docs/REGIMES.md` §2
    measures both, and Gaussian pairs, at 2000 replications. What is pinned
    here is the level the test claims for itself.

    **`|rho| <= 0.5` only**: the test over-rejects at `|rho| = 0.9` for
    `T <= 500` (`.142` in their own table), which is the paper's finding,
    not a defect here.
    """
    n, reps, want = 500, 200, 0.05
    for rho in (0.0, 0.5):
        flags = 0
        for r in range(reps):
            out = run(pair(n, rho, seed=1000 + r), span_rows=n, alpha_adjust="none")
            flags += int(out["flag"][n - 1])
        size = flags / reps
        se = (want * (1 - want) / reps) ** 0.5
        assert abs(size - want) < 4 * se + 0.02, (rho, size, want)
        # Ten flags expected: a monitor that never flags is not "near
        # nominal", and the band above admits zero.
        assert flags >= 2, (rho, flags)


def test_the_power_is_at_least_the_papers():
    """Their Table 2: a `0.5 -> 0.7` break at `T/2` rejects `.587` of the
    time at `T = 500`.

    One-sided, and on Gaussian pairs, where this implementation is well
    above their figure (`.830` at 1000 replications). Under either reading
    of their `t_5` it is within 0.034 of it, `.582` size-adjusted for the
    shared-scale draw, measured in `docs/REGIMES.md` §3.
    """
    n, reps, want = 500, 120, 0.587
    flags = 0
    for r in range(reps):
        a = pair(n // 2, 0.5, seed=7000 + r)
        b = pair(n // 2, 0.7, seed=9000 + r)
        out = run(pl.concat([a, b]), span_rows=n, alpha_adjust="none")
        flags += int(out["flag"][n - 1])
    power = flags / reps
    se = (want * (1 - want) / reps) ** 0.5
    assert power > want - 4 * se, (power, want)
    # And "well above" it, as measured: a floor of 0.7 where 0.83 is the
    # rate at 1000 replications. The paper's figure less 4 se is 0.41.
    assert power >= 0.7, (power, want)


def test_the_scalar_form_tests_the_equicorrelations_level():
    """One statistic however many columns, and it is a **mean** CUSUM: the
    equicorrelation is already one number, so a break in its level is what
    there is to find."""
    cols = [f"x{i}" for i in range(6)]
    kw = {
        "features": cols,
        "span_rows": 400,
        "scalar": True,
        "half_life": 200.0,
        "alpha_adjust": "none",
    }
    steady = run(pair(500, 0.4, seed=20, k=6), **kw)
    assert steady["stat"].drop_nulls().len() == 1, "one statistic per span"
    assert steady["flag"].drop_nulls().to_list() == [False]
    broken = run(pl.concat([pair(250, 0.1, seed=21, k=6), pair(250, 0.9, seed=22, k=6)]), **kw)
    assert broken["flag"].drop_nulls().to_list() == [True]
    assert broken["stat"].drop_nulls()[0] > steady["stat"].drop_nulls()[0]


def test_since_flag_counts_the_rows():
    """The break sits mid-span, at row 150, so the second span straddles it
    and its flag is the break's. On a span boundary, at row 200, no span
    held both regimes: the one flag fell on the last span, not the break's,
    and the restart check sat behind a condition that was false (review
    2026-10-05, TB4)."""
    a = pair(150, 0.1, seed=4)
    b = pair(250, 0.95, seed=5)
    out = run(pl.concat([a, b]), span_rows=100, alpha_adjust="none")
    rows = out.select("stat", "flag", "since_flag").drop_nulls()
    assert rows.height == 4
    # Before any flag, the count runs from the first row.
    assert rows["since_flag"][0] == 100
    # The span holding the break flags, and the count restarts after it.
    assert rows["flag"].to_list()[:2] == [False, True], rows
    assert rows["since_flag"].to_list()[1:3] == [200, 100], rows


# --- the window --------------------------------------------------------------


def test_the_window_statistic_is_the_norm_of_the_difference():
    w = 60
    a = pair(w, 0.1, seed=6, k=3)
    b = pair(w, 0.9, seed=7, k=3)
    df = pl.concat([a, b])
    out = run(
        df,
        features=["x0", "x1", "x2"],
        kind="window",
        span_rows=w,
        crit=0.5,
    )
    pre = np.corrcoef(a.to_numpy().T)
    post = np.corrcoef(b.to_numpy().T)
    iu = np.triu_indices(3, 1)
    want = np.abs(pre[iu] - post[iu]).sum()
    assert out["stat"][2 * w - 1] == pytest.approx(want, rel=1e-12)
    assert out["flag"][2 * w - 1] is True


def test_the_linf_norm_is_the_largest_pair():
    w = 50
    df = pl.concat([pair(w, 0.1, seed=8, k=3), pair(w, 0.9, seed=9, k=3)])
    cols = ["x0", "x1", "x2"]
    l1 = run(df, features=cols, kind="window", span_rows=w, crit=9.0)
    li = run(df, features=cols, kind="window", span_rows=w, crit=9.0, norm="linf")
    assert li["stat"][2 * w - 1] < l1["stat"][2 * w - 1]


def test_the_permutation_critical_value_separates_noise_from_a_break():
    """The flag rate **per row** is not `alpha`: two adjacent windows that
    slide by one row are almost the same windows, so a statistic above the
    quantile stays above it for a run of rows. What the null buys is the
    separation — measured at 9% of rows on a stationary stream against 27%
    on a broken one, with the critical value redrawn every 10 rows."""
    w = 50
    common = {
        "kind": "window",
        "span_rows": w,
        "n_perm": 100,
        "permute_every_rows": 10,
        "features": ["x0", "x1"],
    }
    steady = run(pair(400, 0.4, seed=10), **common)
    quiet = steady["flag"].drop_nulls().to_list()
    broken = run(pl.concat([pair(200, 0.0, seed=11), pair(200, 0.9, seed=12)]), **common)
    loud = broken["flag"].drop_nulls().to_list()
    assert sum(quiet) / len(quiet) < 0.15, sum(quiet) / len(quiet)
    assert sum(loud) / len(loud) > 2 * sum(quiet) / len(quiet)


@pytest.mark.parametrize("every", [1, 3, 10])
def test_the_permutation_critical_value_is_redrawn_every_permute_every_rows_reports(every):
    """S1 (docs/PLAN.md task 196): ``permute_every_rows = n`` redraws the
    critical value every ``n`` reports, as the docstring says. It was drawn
    every ``n + 1``, so ``1`` redrew every other row. Read off the ``crit``
    field: past the first ``2 * span_rows`` rows, a run of ``n`` reports
    holds one value, and the next run another."""
    w = 20
    out = run(
        pair(2 * w + 12 * every, 0.3, seed=196),
        kind="window",
        span_rows=w,
        n_perm=50,
        permute_every_rows=every,
    )
    crit = out["crit"].to_list()
    assert all(c is None for c in crit[: 2 * w]), crit[: 2 * w]
    drawn = crit[2 * w :]
    runs = [len(list(g)) for _, g in itertools.groupby(drawn)]
    assert runs[:-1] == [every] * (len(runs) - 1), runs
    assert runs[-1] <= every, runs


def test_reset_empties_the_windows_at_a_flag():
    w = 40
    df = pl.concat([pair(80, 0.0, seed=13), pair(200, 0.95, seed=14)])
    common = {"kind": "window", "span_rows": w, "crit": 0.4, "features": ["x0", "x1"]}
    kept = run(df, **common)
    cleared = run(df, reset_on_flag=True, **common)
    # After a flag the reset run has no statistic until both windows refill.
    first = kept["flag"].to_list().index(True)
    assert cleared["stat"][first + 1] is None
    assert kept["stat"][first + 1] is not None


# --- the shared contract -----------------------------------------------------


KIND_KW = {
    "monitor": {"span_rows": 150},
    "window": {"kind": "window", "span_rows": 50, "n_perm": 30},
    "sequential": {
        "kind": "sequential",
        "span_rows": 100,
        "monitor_rows": 150,
        "boundary_gamma": 0.25,
    },
}


@pytest.mark.parametrize("kind", ["monitor", "window", "sequential"])
@pytest.mark.parametrize("size", [1, 13, 300])
def test_chunk_invariance(kind, size):
    df = pair(600, 0.4, seed=15)
    kw = KIND_KW[kind]
    want = run(df, **kw)
    bank = po.ModelBank([spec(**kw)])
    got = pl.concat([bank.fit_predict(df[i : i + size]) for i in range(0, df.height, size)])
    assert want.equals(got["c"].struct.unnest())


@pytest.mark.parametrize("kind", ["monitor", "sequential"])
@pytest.mark.parametrize("cut", [60, 170, 300])
def test_save_load_mid_stream(kind, cut):
    """A state saved in a span, in a history or in a monitoring period goes
    on as the unbroken stream does; the sequential kind's critical value is
    recomputed on load, not stored."""
    df = pair(600, 0.4, seed=16)
    s = spec(**KIND_KW[kind])
    want = run(df, **KIND_KW[kind])
    bank = po.ModelBank([s])
    bank.fit_predict(df[:cut])
    again = po.ModelBank.load_bytes(bank.save_bytes(), [s])
    got = again.fit_predict(df[cut:])["c"].struct.unnest()
    assert want[cut:].equals(got)


def test_a_capped_gap_abandons_the_span():
    """Task 47's signal: a span that straddles a break in the clock is not
    a span, so the ring is emptied and the statistic is deferred."""
    df = pair(300, 0.4, seed=17).with_columns(t=pl.int_range(pl.len()).cast(pl.Float64))
    kw = {"span_rows": 100, "clock": "t", "gap_cap": 5.0}
    plain = run(df, **kw)
    gapped = run(
        df.with_columns(
            t=pl.when(pl.int_range(pl.len()) >= 50).then(pl.col("t") + 500.0).otherwise(pl.col("t"))
        ),
        **kw,
    )
    assert plain["stat"][99] is not None
    assert gapped["stat"][99] is None, "the span was abandoned at the gap"
    assert gapped["stat"][149] is not None, "and a new one closed 100 rows later"


def test_a_zero_weight_row_is_not_a_row_of_the_span():
    df = pair(300, 0.4, seed=18).with_columns(w=pl.lit(1.0))
    want = run(df, span_rows=100, weight="w")
    padded = pl.concat([df[:50], df[:1].with_columns(x0=pl.lit(1e6), w=pl.lit(0.0)), df[50:]])
    got = run(padded, span_rows=100, weight="w")
    assert got["stat"][100] == want["stat"][99]


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"features": ["x0"]}, "at least two columns"),
        ({"span_rows": 4}, "span_rows of at least 8"),
        ({"span_rows": None}, "needs `span_rows`"),
        ({"kind": "nope"}, "unknown corrchange kind"),
        ({"alpha": 1.5}, "alpha must be strictly between"),
        ({"alpha_adjust": "nope"}, "unknown alpha_adjust"),
        ({"kind": "window", "span_rows": None}, "needs `span_rows`"),
        ({"kind": "window", "span_rows": 2}, "span_rows of at least 3"),
        ({"kind": "window", "span_rows": 20, "n_perm": 2}, "n_perm >= 20"),
        ({"kind": "window", "span_rows": 20, "scalar": True}, "scalar applies to"),
        ({"half_life": 100.0}, "apply to corrchange only with scalar"),
        ({"emit_sigma": True}, "does not apply to corrchange"),
        # docs/REVIEW-E54-E64.md CC1 and CC3: a critical value that never
        # flags or always does, and the permutation knobs.
        ({"kind": "window", "crit": float("nan")}, "crit must not be NaN"),
        ({"kind": "window", "crit": 0.0}, "crit is the critical value"),
        ({"kind": "sequential", "crit": -1.0}, "crit is the critical value"),
        ({"kind": "window", "crit": float("inf")}, "crit must be finite"),
        ({"bandwidth": 0}, "bandwidth must be >= 1"),
        (
            {"kind": "window", "span_rows": 20, "perm_block": 0},
            "perm_block must be 1..=",
        ),
        (
            {"kind": "window", "span_rows": 20, "perm_block": 21},
            "perm_block must be 1..=",
        ),
        (
            {"kind": "window", "span_rows": 20, "permute_every_rows": 0},
            "permute_every_rows must be >= 1",
        ),
    ],
)
def test_a_bad_spec_is_refused_by_name(kw, message):
    with pytest.raises(ValueError, match=message):
        spec(**kw)


@pytest.mark.parametrize(
    ("kind", "kw", "applies_to"),
    [
        ("monitor", {"crit": 2.0}, '"window" or "sequential"'),
        ("monitor", {"reset_on_flag": True}, '"window"'),
        ("monitor", {"n_perm": 50}, '"window"'),
        ("monitor", {"seed": 3}, '"window"'),
        ("monitor", {"norm": "linf"}, '"window"'),
        ("monitor", {"perm_block": 2}, '"window"'),
        ("monitor", {"permute_every_rows": 10}, '"window"'),
        ("monitor", {"monitor_rows": 50}, '"sequential"'),
        ("monitor", {"boundary_gamma": 0.2}, '"sequential"'),
        ("window", {"bandwidth": 3}, '"monitor" or "sequential"'),
        # "window"'s permutation quantile is taken at `alpha`, so a spread over
        # the pairs changed nothing there (task 160, CD4).
        ("window", {"alpha_adjust": "none"}, '"monitor" or "sequential"'),
        ("window", {"monitor_rows": 50}, '"sequential"'),
        ("window", {"boundary_gamma": 0.2}, '"sequential"'),
        ("sequential", {"n_perm": 50}, '"window"'),
        ("sequential", {"reset_on_flag": True}, '"window"'),
        ("sequential", {"norm": "linf"}, '"window"'),
        ("sequential", {"seed": 1}, '"window"'),
    ],
)
def test_a_parameter_of_another_kind_is_refused(kind, kw, applies_to):
    """A parameter that belongs to another kind is refused, naming the kinds
    it applies to. They were taken and ignored: a ``crit`` given to
    ``"monitor"`` changed nothing, and the docstring's promise that
    ``reset`` under ``"monitor"`` raises was not kept (task 114)."""
    param = next(iter(kw))
    with pytest.raises(ValueError, match=f"{param} applies to kind = {applies_to}"):
        spec(kind=kind, span_rows=50, **kw)


@pytest.mark.parametrize(
    "kw", [{"boundary_gamma": 0.5}, {"boundary_gamma": 0.495}, {"boundary_gamma": -0.1}]
)
def test_the_boundary_exponent_is_at_most_0_49(kw):
    """W&G allow up to 1/2, but the critical value is solved for, and the
    solve's work grows as 1/(1/2 - gamma): minutes to hours past 0.49
    (task 160, CD1)."""
    with pytest.raises(ValueError, match=r"boundary_gamma must be in \[0, 0.49\]"):
        spec(kind="sequential", span_rows=50, **kw)


def test_the_monitoring_period_is_at_least_two_rows():
    with pytest.raises(ValueError, match="monitor_rows of at least 2"):
        spec(kind="sequential", span_rows=50, monitor_rows=1)


# --- the sequential detector (Wied & Galeano 2013) ---------------------------


def _sup_abs_bm_cdf(x):
    """``P(sup_{0<=s<=1} |W(s)| <= x)``, the series (Feller 1951), written
    from the formula rather than read from the library."""
    k = np.arange(400)
    n = 2 * k + 1
    return float(4 / np.pi * np.sum((-1.0) ** k / n * np.exp(-(n**2) * np.pi**2 / (8 * x * x))))


def _sup_abs_bm_quantile(p):
    lo, hi = 0.0, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _sup_abs_bm_cdf(mid) < p else (lo, mid)
    return 0.5 * (lo + hi)


def _sequential_crit(m, ratio, gamma=0.0, **kw):
    out = run(
        pair(m + 5, 0.3),
        kind="sequential",
        span_rows=m,
        monitor_rows=round(ratio * m),
        boundary_gamma=gamma,
        alpha_adjust="none",
        **kw,
    )
    crits = out["crit"].drop_nulls().unique()
    assert crits.len() == 1, crits
    return crits[0]


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 4.0])
def test_the_sequential_critical_value_at_gamma_zero_is_the_series(ratio):
    """Wied & Galeano's Eq. 7 at ``γ = 0``: ``sqrt(T/(1+T))`` times the 95 %
    point of ``sup|W|`` on ``[0, 1]``, 2.2414, with ``T = monitor_rows /
    span_rows``."""
    want = (ratio / (1 + ratio)) ** 0.5 * _sup_abs_bm_quantile(0.95)
    assert _sequential_crit(100, ratio) == pytest.approx(want, abs=1e-6)
    assert _sup_abs_bm_quantile(0.95) == pytest.approx(2.2414, abs=1e-4)


#: Wied & Galeano's Table 1: 5 % critical values, simulated from 10,000
#: paths on a grid of 10,000 points, for T = 0.5, 1, 2, 4.
TABLE_1 = {
    0.0: (1.2870, 1.5578, 1.8158, 1.9980),
    0.25: (1.8001, 1.9924, 2.1684, 2.2467),
    0.45: (2.6282, 2.6844, 2.7215, 2.7660),
}


@pytest.mark.parametrize("gamma", sorted(TABLE_1))
def test_the_sequential_critical_values_are_wied_and_galeanos_table_1(gamma):
    """Within 0.035 of every cell. Theirs sit below ours in 11 of 12, since a
    grid reads the supremum low; the twelfth is above by 0.015, in a row of
    theirs off the exact ``T``-scaling by 0.027, the spread 10,000 paths
    leave."""
    for ratio, theirs in zip((0.5, 1.0, 2.0, 4.0), TABLE_1[gamma], strict=True):
        ours = _sequential_crit(100, ratio, gamma)
        assert abs(ours - theirs) < 0.035, (gamma, ratio, ours, theirs)


def test_the_boundary_law_matches_a_simulation():
    """A second opinion on the solve above ``γ = 0``, independent of it:
    20,000 Brownian paths on a grid of 4,000 points, the 95 % point of
    ``sup |W(s)| / s^0.25``. A grid reads the supremum low (by about
    ``0.58 / sqrt(4000)`` at ``γ = 0``) and the quantile of 20,000 draws is
    good to about 0.015, so the solve sits above the simulation, by no
    more than 0.05."""
    gamma, n, batch = 0.25, 4000, 2000
    rng = np.random.default_rng(114)
    s = np.arange(1, n + 1) / n
    sups = []
    for _ in range(10):
        w = np.cumsum(rng.standard_normal((batch, n)), axis=1) / np.sqrt(n)
        sups.append(np.max(np.abs(w) / s**gamma, axis=1))
    simulated = float(np.quantile(np.concatenate(sups), 0.95))
    ours = _sequential_crit(100, 1.0, gamma) / 0.5 ** (0.5 - gamma)
    assert simulated - 0.015 < ours < simulated + 0.05, (ours, simulated)


def test_nothing_is_reported_in_the_history_or_on_the_first_monitored_row():
    out = run(pair(260, 0.3), kind="sequential", span_rows=100, monitor_rows=100, crit=1e9)
    live = [i for i, v in enumerate(out["stat"].to_list()) if v is not None]
    # History 0..99, first monitored row 100 silent, 101..199 report, then a
    # new history from row 200.
    assert live == list(range(101, 200)), live[:5]


def test_the_sequential_size_is_near_nominal_on_gaussian_pairs():
    """Under the null the share of monitoring periods that end in a flag is
    the size. On about 600 cycles of i.i.d. Gaussian pairs, ``m = 250``,
    ``T = 1``, ``γ = 0``, at 5 %: a rate from 600 cycles has a standard
    error of 0.009."""
    m = 250
    out = run(pair(300_000, 0.4, seed=31), kind="sequential", span_rows=m, alpha_adjust="none")
    live = out["stat"].is_not_null()
    periods = (live & ~live.shift(1, fill_value=False)).sum()
    flags = out["flag"].fill_null(False).sum()
    size = flags / periods
    assert periods > 500
    assert 0.025 < size < 0.075, (flags, periods, size)


def _break(m, before, after, rho0, rho1, seed):
    return pl.concat([pair(m + before, rho0, seed=seed), pair(after, rho1, seed=seed + 1)])


def test_a_sequential_break_is_flagged_and_dated():
    """History and 200 monitored rows at 0.3, then 0.8: a flag after the
    break, dated by Eq. 8 to within 50 rows of it."""
    m = 300
    out = run(_break(m, 200, 400, 0.3, 0.8, 7), kind="sequential", span_rows=m, monitor_rows=600)
    flagged = out.with_row_index("i").filter(pl.col("flag").fill_null(False))
    assert flagged.height >= 1
    first = flagged.row(0, named=True)
    tau = first["i"] - m + 1  # the flag's place in the monitoring period
    assert tau > 200, tau
    assert abs((tau - first["since_change"]) - 200) <= 50, first


def test_a_larger_boundary_exponent_finds_an_early_change_sooner():
    """The trade ``boundary_gamma`` makes, as the paper measured it: a
    change soon after the history (here 10 rows in) is flagged sooner at
    0.45 than at 0, over twenty streams."""
    m = 250
    delays = {0.0: [], 0.45: []}
    for seed in range(20):
        df = _break(m, 10, 490, 0.2, 0.8, 100 + 2 * seed)
        for gamma in delays:
            out = run(
                df, kind="sequential", span_rows=m, monitor_rows=500, boundary_gamma=gamma
            ).with_row_index("i")
            hit = out.filter(pl.col("flag").fill_null(False))
            delays[gamma].append(hit["i"][0] - m - 10 if hit.height else 500)
    assert np.median(delays[0.45]) < np.median(delays[0.0]), delays


def test_since_change_is_null_except_on_a_flag():
    for kind, kw in KIND_KW.items():
        out = run(_break(200, 150, 250, 0.1, 0.9, 3), **kw)
        flags = out["flag"].fill_null(False)
        dated = out["since_change"]
        assert dated.filter(~flags).null_count() == (~flags).sum(), kind
        assert flags.any(), kind
        assert dated.filter(flags).drop_nulls().min() >= 1, kind


# --- the units of the data ---------------------------------------------------


def test_the_monitor_statistic_is_free_of_the_columns_units():
    """A correlation is scale-free, so the monitor's statistic and its
    critical value must not move when one column is rescaled. The
    delta-method gradient carried ``σ_y²`` and ``σ_x²`` in the wrong
    places, so on anything but unit-variance columns ``D̂`` -- and with it
    ``Q`` -- depended on the data's units, and the paper's tables were
    reproduced only where the defect vanished (review 2026-09-18, S3)."""
    n = 300
    df = pair(n, 0.4, seed=19)
    scaled = df.with_columns(pl.col("x1") * 100.0)
    a = run(df, span_rows=n, alpha_adjust="none")
    b = run(scaled, span_rows=n, alpha_adjust="none")
    assert a["stat"][n - 1] is not None, "the span did not close"
    assert b["stat"][n - 1] == pytest.approx(a["stat"][n - 1], rel=1e-9)
    assert b["crit"][n - 1] == a["crit"][n - 1]
    assert b["flag"][n - 1] == a["flag"][n - 1]


def test_the_size_is_near_nominal_on_columns_of_different_scale():
    """The size test above, on ``(x, 100·y)``: the level a correlation test
    claims cannot depend on the units the columns arrive in (S3)."""
    n, reps, want = 500, 200, 0.05
    flags = 0
    for r in range(reps):
        df = pair(n, 0.5, seed=3000 + r).with_columns(pl.col("x1") * 100.0)
        out = run(df, span_rows=n, alpha_adjust="none")
        flags += int(out["flag"][n - 1])
    size = flags / reps
    se = (want * (1 - want) / reps) ** 0.5
    assert abs(size - want) < 4 * se + 0.02, (size, want)
    # The wrong gradient inflated `D̂` by ~1e4 here, so `Q` never reached the
    # critical value and the size read zero -- inside the band above.
    assert flags >= 2, flags
