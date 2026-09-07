"""`marginal(lags=)`: lagged pair moments and a serial-dependence-honest `n`.

`marginal`'s `t` uses `n_kish`, which is the right count for unequal weights
and says nothing about serial dependence. On a smooth stream consecutive rows
are nearly the same observation, so `t` is over-dispersed and reports
significance that is not there. `n_serial` divides by Bartlett's factor
`1 + 2·Σ ρ_x(ℓ)·ρ_y(ℓ)` and `t_serial` is the statistic against it
(`docs/MARGINAL-LAGS-AND-BINS.md`, E66).
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import polars_online as po

LAGS = [1, 2, 3, 5, 8, 13]


def ar1(n, phi, rng):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi * phi) * rng.standard_normal()
    return x


def stream(n=2000, phi_x=0.9, phi_y=0.8, seed=0, link=None):
    rng = np.random.default_rng(seed)
    x, y = ar1(n, phi_x, rng), ar1(n, phi_y, rng)
    if link == "x_follows_y":  # x_t is y_{t-1} plus noise: the feature lags
        x = np.concatenate([[0.0], y[:-1]]) + 0.3 * rng.standard_normal(n)
    if link == "y_follows_x":  # y_t is x_{t-1} plus noise: the feature leads
        y = np.concatenate([[0.0], x[:-1]]) + 0.3 * rng.standard_normal(n)
    return pl.DataFrame({"t": np.arange(n).astype(float), "x": x, "y": y})


def eventful(n=900, seed=23, nulls=True):
    """A stream with every event the ring has to survive: unequal weights
    with zeros, a session change at row 400, a clock gap past ``max_dclock``
    at row 600 and, with ``nulls``, null targets and a null feature (a row
    the model skips)."""
    rng = np.random.default_rng(seed)
    x, y = ar1(n, 0.9, rng), ar1(n, 0.8, rng)
    t = np.arange(n, dtype=float)
    t[600:] += 1e4
    w = rng.uniform(0.2, 2.0, n)
    w[::9] = 0.0
    df = pl.DataFrame({"t": t, "x": x, "y": y, "w": w, "s": ["a"] * 400 + ["b"] * (n - 400)})
    if nulls:
        i = pl.int_range(pl.len())
        df = df.with_columns(
            y=pl.when(i % 7 == 3).then(None).otherwise(pl.col("y")),
            x=pl.when(i % 23 == 5).then(None).otherwise(pl.col("x")),
        )
    return df


EVENTS = dict(
    weight="w",
    clock="t",
    session="s",
    session_gap=1.0,
    max_dclock=100.0,
    halflife=40.0,
    min_periods=5.0,
)


def pairs(df, chunks=1, **kw):
    d = dict(
        targets=["y"],
        features=["x"],
        clock="t",
        halflife=float("inf"),
        max_dclock=1e12,
        min_periods=2.0,
        lags=LAGS,
    )
    d.update(kw)
    bank = po.ModelBank([po.spec.marginal("m", **d)])
    for part in [df] if chunks == 1 else list(df.iter_slices(max(1, len(df) // chunks))):
        bank.fit_predict(part)
    return bank.marginal("m").row(0, named=True)


def test_t_claims_significance_that_t_serial_does_not():
    """Two independent AR(1) series: the true correlation is zero, so a
    statistic that respects serial dependence should not be large."""
    row = pairs(stream(seed=3), serial_rule="geometric")
    assert abs(row["t"]) > 2.0, "the uncorrected statistic should look significant"
    assert abs(row["t_serial"]) < 2.0
    assert row["n_serial"] < row["n_kish"] / 3, "the correction should bite hard at phi=0.9/0.8"


def test_the_geometric_fit_recovers_the_decays():
    row = pairs(stream(seed=5, phi_x=0.9, phi_y=0.8), serial_rule="geometric")
    assert row["phi_x"] == pytest.approx(0.9, abs=0.06)
    assert row["phi_y"] == pytest.approx(0.8, abs=0.06)


def test_the_autocorrelations_decay_like_the_process():
    row = pairs(stream(seed=7, phi_x=0.9, phi_y=0.5))
    for lag, got in zip(LAGS, row["lagcorr_xx"], strict=True):
        assert got == pytest.approx(0.9**lag, abs=0.08), f"lag {lag}"
    for lag, got in zip(LAGS, row["lagcorr_yy"], strict=True):
        assert got == pytest.approx(0.5**lag, abs=0.08), f"lag {lag}"


def test_the_cross_correlations_say_which_series_leads():
    """`lagcorr_yx` is the target now against the feature ℓ back, so a feature
    that *leads* shows there; a feature that follows shows in `lagcorr_xy`."""
    leads = pairs(stream(seed=11, link="y_follows_x"))
    assert leads["lagcorr_yx"][0] > leads["corr"], "a leading feature"
    follows = pairs(stream(seed=11, link="x_follows_y"))
    assert follows["lagcorr_xy"][0] > follows["corr"], "a following feature"


def test_one_chunk_and_many_agree():
    df = stream(n=800, seed=13)
    one, many = (
        pairs(df, chunks=1, serial_rule="geometric"),
        pairs(df, chunks=40, serial_rule="geometric"),
    )
    for key in ("corr", "n_serial", "t_serial", "phi_x", "phi_y"):
        assert one[key] == many[key], key
    for key in ("lagcorr_xx", "lagcorr_yy", "lagcorr_xy", "lagcorr_yx"):
        assert one[key] == many[key], key


@pytest.mark.parametrize("size", [1, 37, 400])
def test_one_chunk_and_many_agree_through_every_event(size):
    """Decay, weights with zeros, null targets, a skipped row, a session
    change and a capped gap -- each moves the ring or the moments, and none
    may notice where a chunk ends."""
    df = eventful()
    s = po.spec.marginal(
        "m", targets=["y"], features=["x"], lags=LAGS, serial_rule="geometric", **EVENTS
    )
    ref = po.ModelBank([s])
    one = ref.fit_predict(df)
    bank = po.ModelBank([s])
    many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)])
    assert many.equals(one, null_equal=True)
    assert bank.marginal("m").equals(ref.marginal("m"), null_equal=True)


def test_the_lagged_pair_is_ew_cov_lags_to_the_bit():
    """The docstring's promise, on the plumbing: through weights, a session
    change and a capped gap, every lagged correlation of the pair is the one
    ``ew_cov(lags=)`` reports for the same two columns -- unclamped, over the
    same two standard deviations, so to the bit and not to a tolerance."""
    df = eventful(nulls=False)
    bank = po.ModelBank([po.spec.marginal("m", targets=["y"], features=["x"], lags=LAGS, **EVENTS)])
    # `ew_cov` reports before each row; the pair is read after the last
    # row, so it is compared with `ew_cov`'s last-row value over one row more.
    bank.fit_predict(df.head(-1))
    pair = bank.marginal("m").row(0, named=True)
    cov = po.spec.ew_cov("c", features=["x", "y"], lags=LAGS, stats=["lagcorr"], **EVENTS)
    last = po.ModelBank([cov]).fit_predict(df)["c"].to_list()[-1]
    for i, lag in enumerate(LAGS):
        assert pair["lagcorr_xx"][i] == last[f"lagcorr_x_x_l{lag}"], lag
        assert pair["lagcorr_yy"][i] == last[f"lagcorr_y_y_l{lag}"], lag
        assert pair["lagcorr_xy"][i] == last[f"lagcorr_x_y_l{lag}"], lag
        assert pair["lagcorr_yx"][i] == last[f"lagcorr_y_x_l{lag}"], lag
    assert 0.5 < pair["lagcorr_xx"][0] < 1.0, "and the numbers are the process's"


def test_the_ring_clears_on_a_capped_gap_and_a_session_change():
    """Task 47's events, and one that is not: a gap under ``max_dclock``
    leaves the ring alone. At ``halflife=inf`` a clock gap changes nothing
    about the numbers except through the ring, so any difference is the
    clearing and not the decay."""
    base = stream(n=80, seed=29).with_columns(s=pl.lit("m"))

    def run(df, **kw):
        kw.setdefault("max_dclock", 5.0)
        return pairs(df, lags=[1], min_periods=0.0, **kw)["lagcorr_xx"]

    plain = run(base)
    at = pl.int_range(pl.len()) >= 40
    under = base.with_columns(t=pl.when(at).then(pl.col("t") + 4.0).otherwise(pl.col("t")))
    assert run(under) == plain, "a gap under max_dclock is not a break"
    over = base.with_columns(t=pl.when(at).then(pl.col("t") + 50.0).otherwise(pl.col("t")))
    assert run(over, max_dclock=100.0) == plain, "nor is a long gap under a higher ceiling"
    assert run(over) != plain, "a capped gap drops the partner one row back"
    sess = base.with_columns(s=pl.when(at).then(pl.lit("a")).otherwise(pl.lit("m")))
    assert run(sess, session="s", session_gap=1.0) != plain, "so does a session change"


def test_lag_moments_hold_across_null_targets():
    """The promise in ``marglag.rs``: a target absent for a run of rows
    leaves every lagged correlation where it was, as the pair moments are
    left, because decay reaches both through the weight and not on its own.
    The first version aged the lag moments by ``lam`` per missing row on
    top, which took a 0.75 autocorrelation to 0.02 over a hundred rows at a
    halflife of twenty while ``corr`` stood still."""
    df = stream(n=400, seed=31, phi_x=0.9, phi_y=0.8)
    df = df.with_columns(y=pl.when(pl.int_range(pl.len()) >= 300).then(None).otherwise(pl.col("y")))
    # `min_periods=0`: the target's weight ages below the default gate and
    # the point is what the moments do, not that the gate closes over them.
    kw = dict(halflife=20.0, lags=[1, 2, 3], serial_rule="truncated", min_periods=0.0)
    before = pairs(df[:300], **kw)
    after = pairs(df, **kw)
    assert before["lagcorr_xx"][0] > 0.5
    for key in ("lagcorr_xx", "lagcorr_yy", "lagcorr_xy", "lagcorr_yx", "corr", "var_x", "var_y"):
        assert before[key] == after[key], key
    # `n_kish` is `W^2/Q`, aged by `lam` and `lam^2` a hundred times over, so
    # the count that divides it is equal to rounding and not to the bit.
    assert after["n_serial"] == pytest.approx(before["n_serial"], rel=1e-12)
    assert after["n_eff"] < before["n_eff"] / 20, "while the target's weight has aged"


def test_a_saved_bank_resumes_with_its_ring(tmp_path):
    df = stream(n=600, seed=17)
    whole = pairs(df, serial_rule="geometric")
    spec = po.spec.marginal(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        halflife=float("inf"),
        max_dclock=1e12,
        min_periods=2.0,
        lags=LAGS,
        serial_rule="geometric",
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df[:300])
    bank.save(tmp_path / "m.state")
    resumed = po.ModelBank.load(tmp_path / "m.state")
    resumed.fit_predict(df[300:])
    got = resumed.marginal("m").row(0, named=True)
    assert got["corr"] == whole["corr"]
    assert got["lagcorr_xx"] == whole["lagcorr_xx"], "the ring must survive the round trip"


def test_without_lags_the_frame_is_what_it_was():
    df = stream(n=300, seed=19)
    bank = po.ModelBank(
        [
            po.spec.marginal(
                "m",
                targets=["y"],
                features=["x"],
                clock="t",
                halflife=float("inf"),
                max_dclock=1e12,
                min_periods=2.0,
            )
        ]
    )
    bank.fit_predict(df)
    cols = bank.marginal("m").columns
    assert not any(c.startswith(("lagcorr", "n_serial", "t_serial", "phi_")) for c in cols)


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"lags": [0, 1]}, "lags must be >= 1"),
        ({"lags": [2, 1]}, "strictly increasing"),
        ({"serial_rule": "geometric"}, "serial_rule needs `lags`"),
        ({"lags": [1], "serial_rule": "nope"}, "unknown serial_rule"),
    ],
)
def test_a_bad_lag_spec_is_refused_by_name(kw, msg):
    d = dict(
        targets=["y"],
        features=["x"],
        clock="t",
        halflife=float("inf"),
        max_dclock=1e12,
        min_periods=2.0,
    )
    d.update(kw)
    with pytest.raises(Exception, match=re.escape(msg)):
        po.ModelBank([po.spec.marginal("m", **d)]).fit_predict(stream(50))
