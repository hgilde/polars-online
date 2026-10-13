"""``po.sim.regimes`` on the paths no other test took (task 252): a state on
the boundary of the correlation cone, which Cholesky cannot factor without
a nudge, and the refusals of a state that is not a correlation matrix and of
the shape parameters, each by name."""

from __future__ import annotations

import numpy as np
import pytest

import polars_online as po

TIER = "essential"

ARGS = dict(transition=[[1.0]], n_blocks=2, rows_per_block=200)


def test_a_singular_state_is_factored_with_a_rounding_scale_nudge():
    """Every entry 1: the two series' returns are one draw. The factor of
    a singular matrix needs a jitter (1e-12 on the diagonal), and the
    returns then agree to its square root, a correlation of 1 to rounding."""
    out = po.sim.regimes(2, states=[np.ones((2, 2))], seed=1, **ARGS)
    rows = out["rows"]
    r1 = np.diff(rows["x_1"].to_numpy(), prepend=0.0)
    r2 = np.diff(rows["x_2"].to_numpy(), prepend=0.0)
    assert np.abs(r1 - r2).max() < 1e-5
    assert np.corrcoef(r1, r2)[0, 1] > 1 - 1e-10
    assert out["truth_blocks"]["corr"].to_list() == [[1.0, 1.0, 1.0]] * 2


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        (
            dict(states=[np.array([[1.0, 0.5], [0.4, 1.0]])]),
            "sim.regimes: state 0 is not symmetric",
        ),
        (
            dict(states=[np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])]),
            "sim.regimes: state 0 is not positive semi-definite",
        ),
        (dict(states=[0.3], cycle_rows=0), "sim.regimes: cycle_rows must be >= 1"),
        (
            dict(states=[0.3], scale_state=[0.0, 1.0]),
            "sim.regimes: scale_state must be a scalar or 1 values",
        ),
        (
            dict(states=[0.3], async_rates=[1.0, 0.0]),
            "sim.regimes: async_rates must be > 0",
        ),
    ],
)
def test_a_parameter_that_cannot_be_simulated_is_refused_by_name(kw, match):
    m = len(kw["states"][0]) if np.ndim(kw["states"][0]) else 2
    with pytest.raises(ValueError, match=match):
        po.sim.regimes(m, **ARGS, **kw)
