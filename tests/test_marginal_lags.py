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
