"""E59: has the correlation structure changed?

Two tests with two nulls. The `monitor` kind is Wied, Krämer & Dehling's
closed-sample constancy test, and its critical value is a *published*
distribution — so its size and power are held to their tables rather than to
a number this implementation happened to produce. The `window` kind measures
how big a change is, against a permutation null; that one is held to a
longhand statistic and to behaving on a stationary stream.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po


def pair(n, rho, seed=0, k=2):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 1))
    x = np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.standard_normal((n, k))
    return pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})


def spec(**kw):
    d = {"features": ["x0", "x1"], "horizon": 200}
    d.update(kw)
    return po.spec.corrchange("c", **d)


def run(df, **kw):
    return po.ModelBank([spec(**kw)]).fit_predict(df)["c"].struct.unnest()


# --- the monitor -------------------------------------------------------------


def test_the_critical_value_is_the_kolmogorov_quantile():
    """Computed from the series, not pinned: 1.3581 at 5%, 1.6276 at 1%,
    1.2239 at 10% are the published values it must reproduce."""
    for alpha, want in ((0.05, 1.3581), (0.01, 1.6276), (0.10, 1.2239)):
        out = run(pair(200, 0.4), horizon=200, alpha=alpha, alpha_adjust="none")
        assert out["crit"][199] == pytest.approx(want, abs=1e-3)
    # Bonferroni over the pairs takes a smaller level, so a larger value.
    three = run(
        pair(200, 0.4, k=3),
        features=["x0", "x1", "x2"],
        horizon=200,
        alpha=0.05,
    )
    assert three["crit"][199] > 1.3581


def test_nothing_is_reported_except_where_a_span_closes():
    out = run(pair(250, 0.3), horizon=100)
    live = [i for i, v in enumerate(out["stat"].to_list()) if v is not None]
    assert live == [99, 199], live
    assert out["flag"][99] in (True, False)
    assert out["flag"][50] is None


def test_a_constant_correlation_is_not_flagged_and_a_break_is():
    """The whole point, at a break the test is built to find."""
    n = 500
    steady = run(pair(n, 0.5, seed=1), horizon=n, alpha_adjust="none")
    assert steady["flag"][n - 1] is False, steady["stat"][n - 1]
    a = pair(n // 2, 0.2, seed=2)
    b = pair(n // 2, 0.9, seed=3)
    broken = run(pl.concat([a, b]), horizon=n, alpha_adjust="none")
    assert broken["flag"][n - 1] is True
    assert broken["stat"][n - 1] > broken["crit"][n - 1]


def test_the_size_is_the_papers():
    """Their Table 1 at the 5% level and `T = 500` reads `.040 / .035 /
    .041` for `rho = -0.5 / 0 / 0.5`. **`|rho| <= 0.5` only**: the test
    over-rejects at `|rho| = 0.9` for `T <= 500` (`.142` in their own
    table), which is the paper's finding, not a defect here."""
    n, reps = 500, 200
    for rho, want in ((0.0, 0.035), (0.5, 0.041)):
        flags = 0
        for r in range(reps):
            out = run(pair(n, rho, seed=1000 + r), horizon=n, alpha_adjust="none")
            flags += int(out["flag"][n - 1])
        size = flags / reps
        se = (want * (1 - want) / reps) ** 0.5
        assert abs(size - want) < 4 * se + 0.02, (rho, size, want)


def test_the_power_is_at_least_the_papers():
    """Their Table 2: a `0.5 -> 0.7` break at `T/2` rejects `.587` of the
    time at `T = 500`."""
    n, reps, want = 500, 120, 0.587
    flags = 0
    for r in range(reps):
        a = pair(n // 2, 0.5, seed=7000 + r)
        b = pair(n // 2, 0.7, seed=9000 + r)
        out = run(pl.concat([a, b]), horizon=n, alpha_adjust="none")
        flags += int(out["flag"][n - 1])
    power = flags / reps
    se = (want * (1 - want) / reps) ** 0.5
    assert power > want - 4 * se - 0.05, (power, want)


def test_the_scalar_form_tests_the_equicorrelations_level():
    """One statistic however many columns, and it is a **mean** CUSUM: the
    equicorrelation is already one number, so a break in its level is what
    there is to find."""
    cols = [f"x{i}" for i in range(6)]
    kw = {
        "features": cols,
        "horizon": 400,
        "scalar": True,
        "halflife": 200.0,
        "alpha_adjust": "none",
    }
    steady = run(pair(500, 0.4, seed=20, k=6), **kw)
    assert steady["stat"].drop_nulls().len() == 1, "one statistic per span"
    assert steady["flag"].drop_nulls().to_list() == [False]
    broken = run(pl.concat([pair(250, 0.1, seed=21, k=6), pair(250, 0.9, seed=22, k=6)]), **kw)
    assert broken["flag"].drop_nulls().to_list() == [True]
    assert broken["stat"].drop_nulls()[0] > steady["stat"].drop_nulls()[0]


def test_since_flag_counts_the_rows():
    n = 400
    a = pair(n // 2, 0.1, seed=4)
    b = pair(n // 2, 0.95, seed=5)
    out = run(pl.concat([a, b]), horizon=100, alpha_adjust="none")
    rows = out.select("stat", "flag", "since_flag").drop_nulls()
    assert rows.height == 4
    # Before any flag, the count runs from the first row.
    assert rows["since_flag"][0] == 100
    flagged = [i for i, f in enumerate(rows["flag"].to_list()) if f]
    assert flagged, rows
    # And it restarts after one.
    if flagged[0] + 1 < rows.height:
        assert rows["since_flag"][flagged[0] + 1] == 100


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
        window=w,
        horizon=None,
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
    l1 = run(df, features=cols, kind="window", window=w, horizon=None, crit=9.0)
    li = run(df, features=cols, kind="window", window=w, horizon=None, crit=9.0, norm="linf")
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
        "window": w,
        "horizon": None,
        "n_perm": 100,
        "permute_every": 10,
        "features": ["x0", "x1"],
    }
    steady = run(pair(400, 0.4, seed=10), **common)
    quiet = steady["flag"].drop_nulls().to_list()
    broken = run(pl.concat([pair(200, 0.0, seed=11), pair(200, 0.9, seed=12)]), **common)
    loud = broken["flag"].drop_nulls().to_list()
    assert sum(quiet) / len(quiet) < 0.15, sum(quiet) / len(quiet)
    assert sum(loud) / len(loud) > 2 * sum(quiet) / len(quiet)


def test_reset_empties_the_windows_at_a_flag():
    w = 40
    df = pl.concat([pair(80, 0.0, seed=13), pair(200, 0.95, seed=14)])
    common = {"kind": "window", "window": w, "horizon": None, "crit": 0.4, "features": ["x0", "x1"]}
    kept = run(df, **common)
    cleared = run(df, reset=True, **common)
    # After a flag the reset run has no statistic until both windows refill.
    first = kept["flag"].to_list().index(True)
    assert cleared["stat"][first + 1] is None
    assert kept["stat"][first + 1] is not None


# --- the shared contract -----------------------------------------------------


@pytest.mark.parametrize("kind", ["monitor", "window"])
@pytest.mark.parametrize("size", [1, 13, 300])
def test_chunk_invariance(kind, size):
    df = pair(600, 0.4, seed=15)
    kw = (
        {"horizon": 150}
        if kind == "monitor"
        else {"kind": "window", "window": 50, "horizon": None, "n_perm": 30}
    )
    want = run(df, **kw)
    bank = po.ModelBank([spec(**kw)])
    got = pl.concat([bank.fit_predict(df[i : i + size]) for i in range(0, df.height, size)])
    assert want.equals(got["c"].struct.unnest())


def test_save_load_mid_stream():
    df = pair(600, 0.4, seed=16)
    s = spec(horizon=150)
    want = run(df, horizon=150)
    bank = po.ModelBank([s])
    bank.fit_predict(df[:300])
    again = po.ModelBank.load_bytes(bank.save_bytes(), [s])
    got = again.fit_predict(df[300:])["c"].struct.unnest()
    assert want[300:].equals(got)


def test_a_capped_gap_abandons_the_span():
    """Task 47's signal: a span that straddles a break in the clock is not
    a span, so the ring is emptied and the statistic is deferred."""
    df = pair(300, 0.4, seed=17).with_columns(t=pl.int_range(pl.len()).cast(pl.Float64))
    kw = {"horizon": 100, "clock": "t", "max_dclock": 5.0}
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
    want = run(df, horizon=100, weight="w")
    padded = pl.concat([df[:50], df[:1].with_columns(x0=pl.lit(1e6), w=pl.lit(0.0)), df[50:]])
    got = run(padded, horizon=100, weight="w")
    assert got["stat"][100] == want["stat"][99]


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"features": ["x0"]}, "at least two columns"),
        ({"horizon": 4}, "horizon of at least 8"),
        ({"horizon": None}, "needs `horizon`"),
        ({"kind": "nope"}, "unknown corrchange kind"),
        ({"alpha": 1.5}, "alpha must be strictly between"),
        ({"alpha_adjust": "nope"}, "unknown alpha_adjust"),
        ({"kind": "window", "horizon": None}, 'kind = "window" needs `window`'),
        ({"kind": "window", "horizon": None, "window": 2}, "window of at least 3"),
        ({"kind": "window", "horizon": None, "window": 20, "n_perm": 2}, "n_perm >= 20"),
        ({"kind": "window", "horizon": None, "window": 20, "scalar": True}, "scalar applies to"),
        ({"halflife": 100.0}, "apply to corrchange only with scalar"),
        ({"emit_sigma": True}, "does not apply to corrchange"),
    ],
)
def test_a_bad_spec_is_refused_by_name(kw, message):
    with pytest.raises(ValueError, match=message):
        spec(**kw)


def test_the_expression_equals_the_bank():
    df = pair(300, 0.4, seed=19)
    want = run(df, horizon=100)
    with pytest.warns(po.InMemoryExpressionWarning):
        got = df.select(pl.col("x0").online.corrchange(["x1"], horizon=100).alias("c"))[
            "c"
        ].struct.unnest()
    assert want.equals(got)
