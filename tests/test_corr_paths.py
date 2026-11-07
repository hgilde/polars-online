"""``po.corr`` on the paths no other test took (task 252): each refusal of
an argument of the wrong shape, by name; the summaries that are undefined
below two series (``nan``, as the docstrings say); a minimum-variance
weight whose normaliser is zero; and the Epps inversion read from a closed
row rather than a Gram mapping."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import corr

TIER = "essential"

R2 = np.array([[1.0, 0.3], [0.3, 1.0]])


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda: corr.shrink(R2, x=np.zeros((10, 3))),
            r"po.corr.shrink: x must have one column for each of the matrix\'s 2 columns, but it "
            r"has shape \(10, 3\)",
        ),
        (
            lambda: corr.shrink(R2, x=np.zeros(10)),
            r"po.corr.shrink: x must have one column for each of the matrix\'s 2 columns, but it "
            r"has shape \(10,\)",
        ),
        (
            lambda: corr.absorption(R2, 0),
            "po.corr.absorption: k must be from 1 to the matrix's 2 columns, but it is 0",
        ),
        (
            lambda: corr.absorption(R2, 3),
            "po.corr.absorption: k must be from 1 to the matrix's 2 columns, but it is 3",
        ),
        (
            lambda: corr.absorption_shift(np.zeros(3), np.zeros(4)),
            r"po.corr.absorption_shift: ar_fast and ar_slow must have the same shape, but they "
            r"have \(3,\) and \(4,\)",
        ),
        (
            lambda: corr.from_spectral([1.0, 1.0], np.eye(3)),
            r"po.corr.from_spectral: 2 eigenvalues need 2 eigenvectors, one for each row of vecs, "
            r"but vecs has shape \(3, 3\)",
        ),
        (
            lambda: corr.from_blocks(np.eye(3), ["a", "a", "b"]),
            r"po.corr.from_blocks: labels name 2 blocks, so b must be a 2 by 2 matrix, but it has "
            r"shape \(3, 3\)",
        ),
        (
            lambda: corr.mp_edge(0, 10),
            "po.corr.mp_edge: n and m must be above 0, but they are 0 and 10",
        ),
        (
            lambda: corr.mp_edge(10, -1),
            "po.corr.mp_edge: n and m must be above 0, but they are 10 and -1",
        ),
        (
            lambda: corr.signal_share(np.zeros(3), np.ones(2)),
            "po.corr.signal_share: n_kish_blocks must hold one sample size for each of the 3 "
            "blocks",
        ),
        (
            lambda: corr.loss(R2, np.eye(3), "qlike"),
            r"po.corr.loss: fcst and real must have the same shape, but they have \(2, 2\) and "
            r"\(3, 3\)",
        ),
    ],
)
def test_an_argument_of_the_wrong_shape_is_refused_by_name(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_the_equicorrelation_of_fewer_than_two_series_is_nan():
    assert math.isnan(corr.equicorr(np.eye(1)))
    assert math.isnan(corr.equicorr_row([0.4]))
    # Two series: their one correlation.
    assert corr.equicorr(R2) == pytest.approx(0.3, rel=1e-15)


def test_a_minimum_variance_weight_with_a_zero_normaliser_has_no_loss():
    """``w = F^-1 mu / (mu' F^-1 mu)``: an indefinite forecast can make the
    normaliser exactly zero, and the loss is then undefined (``nan``), not
    an infinity or a division error."""
    f = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert math.isnan(corr.loss(f, np.eye(2), "minvar", mu=[1.0, 1.0]))
    # Off the zero, the same forecast has a loss: F^-1 mu = (1, -1/2), the
    # normaliser 3/4, w = (4/3, -2/3), and w' I w = 20/9.
    assert corr.loss(f, np.eye(2), "minvar", mu=[1.0, 0.5]) == pytest.approx(20 / 9, rel=1e-15)


def test_the_epps_inversion_reads_a_closed_row_as_the_gram_it_closed():
    """A group's closed row (``closed_groups``) carries its lagged
    co-moments as the Gram mapping of the open group does: inverted from
    either, the correlation at scale ``L`` is the same, to the bit."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal(600)
    rows = pl.DataFrame({"x0": a, "x1": np.concatenate([[0.0], a[:-1]]) + 0.1 * a})
    spec = po.spec.ew_cov(
        "c",
        features=["x0", "x1"],
        lam=1.0,
        stats=[],
        lags=[1, 2],
        group="g",
        group_close="monotone",
    )
    first = rows.with_columns(g=pl.lit("a"))
    closing = pl.DataFrame({"g": ["b"], "x0": [0.0], "x1": [0.0]})
    bank = po.ModelBank([spec])
    bank.fit_predict(first)
    open_gram = bank.gram("c")[0]
    bank.fit_predict(closing)
    row = bank.closed_groups()
    assert row.height == 1 and row["group"].item() == "a"
    want = corr.epps_invert(open_gram, L=3)
    got = corr.epps_invert(row, L=3)
    np.testing.assert_array_equal(got, want)
    assert got[0, 1] != po.gram.correlation(open_gram)[0, 1], "the lags moved it"
