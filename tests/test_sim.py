"""E64: the regime simulator — data whose truth is known.

Every detector in this library is a claim about a stream, and a claim needs
a stream whose answer is written down. These tests are the simulator's own:
that what it says is true *is* true in the data, and that the awkward parts
— asynchrony, microstructure noise, an AR filter, a volatility state, a
diurnal pattern, a volume clock — each do what they say.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import sim


def two_state(**kw):
    d = dict(
        states=[0.2, 0.7],
        transition=[[0.98, 0.02], [0.02, 0.98]],
        n_blocks=8,
        bars_per_block=500,
        seed=0,
    )
    d.update(kw)
    return sim.regimes(4, **d)


def returns(bars, m=4):
    x = bars.select([f"x_{i + 1}" for i in range(m)]).to_numpy()
    return np.diff(x, axis=0)


def test_the_frames_have_the_documented_schema_and_lengths():
    out = two_state(n_blocks=3, bars_per_block=50)
    assert set(out) == {"bars", "truth_rows", "truth_blocks"}
    bars, rows, blocks = out["bars"], out["truth_rows"], out["truth_blocks"]
    assert bars.height == rows.height == 150
    assert blocks.height == 3
    assert bars.columns == [
        "instrument",
        "t",
        "clock",
        "session",
        "x_1",
        "x_2",
        "x_3",
        "x_4",
        "volume",
    ]
    assert rows.columns == ["t", "block", "state", "vol_mult", "mix"]
    assert blocks.columns == ["block", "state", "n_bars", "corr"]
    assert bars["instrument"].unique().to_list() == ["sim"]
    assert bars["volume"].null_count() == 150, "no volume unless asked for"
    assert blocks["corr"][0].len() == 10, "vech of a 4x4"


def test_a_seed_reproduces_every_frame_byte_for_byte():
    a, b = two_state(n_blocks=3, bars_per_block=40), two_state(n_blocks=3, bars_per_block=40)
    for key in ("bars", "truth_rows", "truth_blocks"):
        assert a[key].equals(b[key]), key
    other = two_state(n_blocks=3, bars_per_block=40, seed=1)
    assert not a["bars"].equals(other["bars"])


def test_the_per_state_correlations_are_recovered():
    """Pooled over the blocks in each state, the sample correlation of the
    returns is the state's matrix, within the Fisher-z sampling floor."""
    out = two_state()
    ret = returns(out["bars"])
    state = out["truth_rows"]["state"].to_numpy()[1:]
    for s, rho in ((0, 0.2), (1, 0.7)):
        rows = ret[state == s]
        if len(rows) < 100:
            continue
        r = np.corrcoef(rows.T)
        off = r[~np.eye(4, dtype=bool)]
        se = po.corr.fisher_se(len(rows), rho=rho)
        assert abs(off.mean() - rho) < 3 * se, (s, off.mean(), rho, se)


def test_the_truth_blocks_are_the_states_matrices():
    out = two_state(n_blocks=4, bars_per_block=100)
    iu = np.triu_indices(4)
    for row in out["truth_blocks"].iter_rows(named=True):
        want = np.full((4, 4), 0.2 if row["state"] == 0 else 0.7)
        np.fill_diagonal(want, 1.0)
        assert np.allclose(row["corr"], want[iu])
        assert row["n_bars"] == 100


def test_durations_make_the_sojourn_exact():
    out = two_state(
        n_blocks=9,
        bars_per_block=20,
        durations=[2, 1],
        transition=[[0.5, 0.5], [0.5, 0.5]],
    )
    seq = out["truth_blocks"]["state"].to_list()
    assert seq == [0, 0, 1, 0, 0, 1, 0, 0, 1], seq


def test_a_matrix_state_is_taken_as_given():
    r = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
    out = sim.regimes(
        3,
        states=[r],
        transition=[[1.0]],
        n_blocks=4,
        bars_per_block=4000,
        seed=2,
    )
    ret = returns(out["bars"], m=3)
    got = np.corrcoef(ret.T)
    assert np.allclose(got, r, atol=0.03), got


def test_smooth_interpolates_and_every_matrix_on_the_way_is_valid():
    out = two_state(
        n_blocks=4, bars_per_block=200, design="smooth", smooth_bars=40, durations=[1, 1]
    )
    mix = out["truth_rows"]["mix"].to_numpy()
    assert mix.max() == pytest.approx(1.0)
    assert (mix == 0.0).sum() > 0.7 * len(mix), "the ramp is local to a boundary"
    # Every interpolated matrix is a convex combination of two correlation
    # matrices, so it is one.
    for f in np.unique(mix):
        r = (1 - f) * np.full((4, 4), 0.2) + f * np.full((4, 4), 0.7)
        np.fill_diagonal(r, 1.0)
        assert np.linalg.eigvalsh(r).min() > -1e-12
    # And the block truth is the *mean* over the block, not the endpoint.
    blocks = out["truth_blocks"]
    ends = {round(v[1], 6) for v in blocks["corr"].to_list()}
    assert len(ends) > 1


def test_phi_is_recovered_from_the_lag_one_autocorrelation():
    out = sim.regimes(
        2,
        states=[0.3],
        transition=[[1.0]],
        n_blocks=1,
        bars_per_block=40_000,
        phi=[0.6, 0.0],
        seed=3,
    )
    ret = returns(out["bars"], m=2)
    for i, want in enumerate((0.6, 0.0)):
        got = np.corrcoef(ret[1:, i], ret[:-1, i])[0, 1]
        assert abs(got - want) < 0.02, (i, got, want)


def test_noise_makes_the_observed_return_an_ma_one():
    """Microstructure noise on the *level* gives a negative first
    autocorrelation of the observed return — the effect `rcov` undoes."""
    common = dict(states=[0.3], transition=[[1.0]], n_blocks=1, bars_per_block=20_000, seed=4)
    clean = returns(sim.regimes(2, noise=0.0, **common)["bars"], m=2)
    noisy = returns(sim.regimes(2, noise=2.0, **common)["bars"], m=2)
    a_clean = np.corrcoef(clean[1:, 0], clean[:-1, 0])[0, 1]
    a_noisy = np.corrcoef(noisy[1:, 0], noisy[:-1, 0])[0, 1]
    assert abs(a_clean) < 0.03
    assert a_noisy < -0.2, a_noisy


def test_the_volatility_state_scales_the_returns():
    out = sim.regimes(
        2,
        states=[0.3, 0.3],
        transition=[[0.5, 0.5], [0.5, 0.5]],
        n_blocks=8,
        bars_per_block=2000,
        durations=[1, 1],
        vol_state=np.log(3.0),
        seed=5,
    )
    ret = returns(out["bars"], m=2)
    state = out["truth_rows"]["state"].to_numpy()[1:]
    sd = [ret[state == s].std() for s in (0, 1)]
    assert sd[1] / sd[0] == pytest.approx(3.0, rel=0.1), sd
    assert np.allclose(
        out["truth_rows"]["vol_mult"].to_numpy(),
        np.where(out["truth_rows"]["state"].to_numpy() == 1, 3.0, 1.0),
    )


def test_the_diurnal_pattern_scales_the_off_diagonal():
    pattern = [0.2, 1.0, 1.0, 1.0]
    out = sim.regimes(
        3,
        states=[0.8],
        transition=[[1.0]],
        n_blocks=1,
        bars_per_block=40_000,
        session_bars=4,
        diurnal=pattern,
        seed=6,
    )
    ret = returns(out["bars"], m=3)
    # Bar `t` of the session produced return `t` (the diff at index t-1
    # spans bars t-1 and t); compare the quiet phase with a busy one.
    t = np.arange(1, len(ret) + 1) % 4
    quiet = np.corrcoef(ret[t == 0].T)[0, 1]
    busy = np.corrcoef(ret[t == 2].T)[0, 1]
    assert busy > quiet + 0.2, (quiet, busy)


def test_the_session_and_the_clock_follow_their_parameters():
    out = two_state(n_blocks=2, bars_per_block=60, session_bars=20, volume=(100.0, 2.0))
    bars = out["bars"]
    assert bars["session"].to_list() == [t // 20 for t in range(120)]
    assert bars["volume"].null_count() == 0
    assert bars["volume"].min() > 0
    assert np.allclose(bars["clock"].to_numpy(), np.cumsum(bars["volume"].to_numpy()))
    # Without a volume the clock is the bar index.
    plain = two_state(n_blocks=1, bars_per_block=10)["bars"]
    assert plain["clock"].to_list() == list(range(10))


def test_async_rates_drop_ticks_and_the_epps_curve_rises():
    """The reason asynchrony is in here: a correlation computed over fine
    intervals of previous-tick prices is attenuated, and recovers as the
    interval grows."""
    out = sim.regimes(
        2,
        states=[0.8],
        transition=[[1.0]],
        n_blocks=1,
        bars_per_block=60_000,
        async_rates=[0.25, 0.25],
        seed=7,
    )
    bars = out["bars"]
    assert bars["x_1"].null_count() > 0, "a bar with no tick carries null"
    filled = bars.select(pl.col("x_1", "x_2").forward_fill()).drop_nulls()
    x = filled.to_numpy()
    curve = []
    for step in (1, 5, 20, 100):
        r = np.diff(x[::step], axis=0)
        curve.append(np.corrcoef(r.T)[0, 1])
    assert curve == sorted(curve), curve
    assert curve[0] < 0.6 and curve[-1] > 0.7, curve


def test_the_output_feeds_refresh_time():
    from polars_online import prep

    out = sim.regimes(
        3,
        states=[0.5],
        transition=[[1.0]],
        n_blocks=1,
        bars_per_block=2000,
        async_rates=[0.5, 0.5, 0.5],
        seed=8,
    )
    long = (
        out["bars"]
        .select("t", "x_1", "x_2", "x_3")
        .unpivot(index="t", variable_name="series", value_name="px")
        .drop_nulls()
        .sort("t")
    )
    grid = prep.refresh_time(
        long, series="series", names=["x_1", "x_2", "x_3"], time="t", value="px"
    ).collect()
    assert grid.height > 100
    assert grid["retained_fraction"].max() <= 1.0


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"states": [1.5]}, "not a correlation matrix"),
        ({"states": [np.eye(3)]}, r"is \(3, 3\), not \(4, 4\)"),
        ({"states": [np.full((4, 4), 0.5)]}, "diagonal that is not 1"),
        ({"transition": [[0.5, 0.4], [0.5, 0.5]]}, "row-stochastic"),
        ({"transition": [[1.0]]}, "not \\(2, 2\\)"),
        ({"design": "nope"}, 'design must be "step" or "smooth"'),
        ({"phi": [2.0, 0.0, 0.0, 0.0]}, r"\|phi\| must be < 1"),
        ({"phi": [0.5]}, "phi must be a scalar or 4 values"),
        ({"durations": [1]}, "one positive block count per state"),
        ({"diurnal": [0.5]}, "one multiplier per session bar"),
        ({"diurnal": [0.0] * 500}, r"must be in \(0, 1\]"),
        ({"async_rates": [1.0]}, "one expected tick rate per series"),
        ({"volume": (0.0, 1.0)}, r"volume is \(mean, shape\)"),
        ({"n_blocks": 0}, "must be >= 1"),
    ],
)
def test_a_bad_call_is_refused_by_name(kw, message):
    opts = {"n_blocks": 2, "bars_per_block": 500, **kw}
    with pytest.raises(ValueError, match=message):
        two_state(**opts)


def test_fewer_than_two_series_is_refused():
    with pytest.raises(ValueError, match="at least two series"):
        sim.regimes(1, states=[0.5], transition=[[1.0]], n_blocks=1, bars_per_block=10)
