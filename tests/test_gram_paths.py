"""``po.gram`` on the paths no other test took (task 252): Gram mappings that
lack the optional target moments (a mapping handed to ``from_row`` may leave
them out, and the functions say ``None`` for what they cannot form), the
within-part merge of Grams with no target, the refusals of ``merge``,
``from_row`` and ``coef_stats``, co-moments handed over as integers, and the
VIFs of a Gram whose every column is constant."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po
from test_gram_fits import SUBSETS, blocks

TIER = "essential"

OPTIONAL = ("target_means", "target_vars", "target_n_kish", "cross_centered", "n_kish")


def _closed_rows(fit_intercept: bool = True, targets: tuple[str, ...] = ("y",)) -> list[dict]:
    """Two closed groups' rows of one spec, as ``iter_rows(named=True)``
    gives them."""
    rng = np.random.default_rng(11)
    n = 60
    df = pl.DataFrame(
        {
            "g": ["a"] * n + ["b"] * n + ["c"],
            "x0": rng.normal(size=2 * n + 1),
            "x1": rng.normal(size=2 * n + 1),
            "y": rng.normal(size=2 * n + 1),
        }
    )
    common = dict(features=["x0", "x1"], half_life=20.0, group="g", group_close="monotone")
    if targets:
        spec = po.spec.ewridge("r", targets=list(targets), fit_intercept=fit_intercept, **common)
    else:
        spec = po.spec.ew_cov("r", **common)
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    rows = list(bank.closed_groups().iter_rows(named=True))
    assert len(rows) == 2
    return rows


def _bare(row: dict) -> po.gram.Gram:
    d = dict(row)
    for k in OPTIONAL:
        d.pop(k)
    return po.gram.from_row(d)


def test_a_mapping_without_the_target_moments_merges_what_it_has():
    """A mapping with the feature sums alone: ``from_row`` reads the missing
    moments as ``None``, and ``merge`` pools the rest exactly as it pools
    the full Grams, and leaves ``None`` where a part had none."""
    rows = _closed_rows()
    bare = po.gram.merge([_bare(r) for r in rows])
    full = po.gram.merge([po.gram.from_row(r) for r in rows])
    for k in OPTIONAL:
        assert bare[k] is None, k
        assert full[k] is not None, k
    for k in ("weight_sum", "means", "comoments", "cross_moments", "means_by_target"):
        np.testing.assert_array_equal(bare[k], full[k], err_msg=k)
    # Without ``cross_centered`` the solve forms the centred cross-moments
    # from the raw ones and the means, another rounding than the
    # accumulator's own: the same fit, to the last bit or so.
    np.testing.assert_allclose(
        po.gram.solve(bare, target="y"), po.gram.solve(full, target="y"), rtol=1e-12
    )
    # A mapping built by hand may lack the means behind each target too.
    g = po.gram.from_row(rows[0])
    merged = po.gram.merge([g, dict(po.gram.from_row(rows[1]), means_by_target=None)])
    assert merged["means_by_target"] is None and merged["cross_centered"] is None
    np.testing.assert_array_equal(merged["comoments"], full["comoments"])


def test_a_within_merge_of_grams_with_no_target_averages_their_comoments():
    """``ew_cov`` keeps no target, so its within-part merge has no
    cross-moments to centre, and its co-moments are the parts' own,
    averaged by their weights."""
    grams = [po.gram.from_row(r) for r in _closed_rows(targets=())]
    pooled = po.gram.merge(grams, within=True)
    w = np.array([g["weight_sum"] for g in grams])
    want = sum(wi * g["comoments"] for wi, g in zip(w, grams, strict=True)) / w.sum()
    np.testing.assert_allclose(pooled["comoments"], want, rtol=1e-14)
    assert pooled["targets"] == [] and pooled["cross_centered"].shape == (0, 2)
    assert pooled["within_parts"] == 2


def test_a_within_merge_centres_the_cross_moments_a_part_lacks():
    """A part without ``cross_centered`` has its cross-moments centred at
    its means, as :func:`solve` forms them; the pooled Gram is the one the
    full parts give, to rounding."""
    rows = _closed_rows()
    bare = po.gram.merge([_bare(r) for r in rows], within=True)
    full = po.gram.merge([po.gram.from_row(r) for r in rows], within=True)
    np.testing.assert_array_equal(bare["comoments"], full["comoments"])
    np.testing.assert_allclose(bare["cross_centered"], full["cross_centered"], rtol=1e-12)


def test_a_within_merge_with_no_way_to_centre_the_cross_moments_is_refused():
    """Through the origin, with neither ``cross_centered`` nor
    ``target_means``, a part's cross-moments about its means cannot be
    formed."""
    grams = [_bare(r) for r in _closed_rows(fit_intercept=False)]
    with pytest.raises(
        ValueError,
        match="merge\\(within=True\\) needs each target's cross-moments about its means: this "
        "part has no cross_centered, and no intercept or target_means to centre them at",
    ):
        po.gram.merge(grams, within=True)


def test_merge_refuses_parts_with_other_targets():
    a = po.gram.from_row(_closed_rows()[0])
    b = dict(a, targets=["z"])
    with pytest.raises(ValueError, match="merge\\(\\) needs the same targets in every part"):
        po.gram.merge([a, b])


def test_from_row_refuses_what_is_not_one_closed_row():
    rows = _closed_rows()
    frame = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="from_row\\(\\) takes one closed row; this frame has 2"):
        po.gram.from_row(frame)
    with pytest.raises(ValueError, match="this frame has 0"):
        po.gram.from_row(frame.clear())
    with pytest.raises(
        TypeError,
        match="from_row\\(\\) takes a one-row frame from ModelBank.closed_groups\\(\\), a row of "
        "its iter_rows\\(named=True\\), or a mapping; got int",
    ):
        po.gram.from_row(42)
    short = dict(rows[0], comoments=rows[0]["comoments"][:-1])
    with pytest.raises(ValueError, match="comoments has 5 entries; a 3-column Gram packs 6"):
        po.gram.from_row(short)
    # One row of the frame reads as the row itself.
    one = po.gram.from_row(frame.head(1))
    np.testing.assert_array_equal(one["comoments"], po.gram.from_row(rows[0])["comoments"])


def test_integer_comoments_are_read_as_float64():
    """The path functions hand the co-moments to the native solver as they
    are when float64 or float32; any other dtype is converted first."""
    g = po.gram.from_row(_closed_rows()[0])
    ints = dict(g, comoments=np.round(g["comoments"] * 1000).astype(np.int64))
    floats = dict(g, comoments=np.round(g["comoments"] * 1000))
    penalties = [1.0, 0.1, 0.01]
    np.testing.assert_array_equal(
        po.gram.lasso_path(ints, penalties, target="y"),
        po.gram.lasso_path(floats, penalties, target="y"),
    )


def test_coef_stats_through_the_origin_needs_the_target_mean():
    g = po.gram.from_row(_closed_rows(fit_intercept=False)[0])
    coef = po.gram.solve(g, target="y")
    po.gram.coef_stats(g, coef, target="y")  # with target_means it answers
    with pytest.raises(
        ValueError,
        match=r"a fit through the origin reads the target's mean square, Var\[y\] \+ "
        r"mean\*\*2, and this Gram has no target_means",
    ):
        po.gram.coef_stats(dict(g, target_means=None), coef, target="y")


def test_the_vif_of_a_gram_of_constant_columns_is_inf_throughout():
    """A constant column has no correlation to read, and a VIF of ``inf``;
    with every column constant there is no column left to invert over."""
    df = pl.DataFrame({"x0": [2.0] * 20, "x1": [-1.0] * 20})
    bank = po.ModelBank([po.spec.ew_cov("c", features=["x0", "x1"], half_life=10.0)])
    bank.fit_predict(df)
    v = po.gram.vif(bank.gram("c")[0])
    assert v.tolist() == [np.inf, np.inf]


def test_a_path_over_subsets_takes_penalty_weights_as_lars_path_does():
    """``solve_subsets(path={"penalty_weights": w})`` is ``lars_path`` with
    the same weights on each subset's merge, to the bit; the weights move
    the path, so a path that ignored them would show."""
    grams = blocks()
    w = [0.5, 1.0, 2.0, 1.0, 4.0]
    got = po.gram.solve_subsets(grams, SUBSETS, path={"max_active": 3, "penalty_weights": w})
    plain = po.gram.solve_subsets(grams, SUBSETS, path={"max_active": 3})
    moved = False
    for subset, per, unweighted in zip(SUBSETS, got, plain, strict=True):
        merged = po.gram.merge([grams[i] for i in subset])
        for t, path, other in zip(["y", "y2"], per, unweighted, strict=True):
            want = po.gram.lars_path(merged, target=t, max_active=3, penalty_weights=w)
            assert np.array_equal(path["coef"], want["coef"])
            assert np.array_equal(path["penalties"], want["penalties"])
            assert path["active"] == want["active"]
            moved |= path["active"] != other["active"] or not np.array_equal(
                path["coef"], other["coef"]
            )
    assert moved


def test_a_float32_gram_names_the_pair_whose_co_moment_overflows():
    """A Gram read in float32 is refused where a co-moment is past
    float32's range: one of two columns is named as the pair, and the fix
    is to rescale both."""
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x0": rng.normal(size=200) * 1e15, "x1": rng.normal(size=200) * 1e30})
    bank = po.ModelBank([po.spec.ew_cov("c", features=["x0", "x1"], half_life=50.0)])
    bank.fit_predict(df)
    with pytest.raises(
        ValueError,
        match=r"gram\(dtype=\"float32\"\): the co-moment of 'x0' and 'x1' is -?\d\.\d{3}e44, "
        "past float32's largest number, 3.4e38, where it reads inf: rescale those columns "
        "upstream, or read the Gram in float64",
    ):
        bank.gram("c", dtype="float32")
    assert np.isfinite(bank.gram("c")[0]["comoments"]).all(), "float64 holds it"
