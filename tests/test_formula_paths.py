"""The window formulas on the paths no other test took (task 252): the
nodes and arguments a formula refuses, each by name. ``po.ewm_mean`` and its
siblings read an input expression's serialized tree
(``expr.meta.serialize``) into a compact tree of their own; anything outside
the element-wise nodes that tree holds is refused, since a node that reads
other rows would read only the chunk's."""

from __future__ import annotations

import polars as pl
import pytest

import polars_online as po
from polars_online._formula import PREFIX, FormulaError, from_tree, to_tree

TIER = "essential"


@pytest.mark.parametrize(
    ("expr", "named"),
    [
        (pl.col("x") + pl.lit(b"ab"), "this literal is not read"),
        (pl.col("x") + pl.lit([1, 2]), "this literal is not read"),
        (pl.col("x") + pl.len(), 'this node is not read ("Len")'),
        (pl.col(PREFIX + "{not json") + 1.0, "names starting with '@po:' are reserved"),
        (pl.col("x") // 2, "the operator 'FloorDivide' is not read"),
        (pl.col("x") % 2, "the operator 'Modulus' is not read"),
        (pl.col("x").cast(pl.Int128), 'a cast to "Int128" is not read'),
        (pl.col("x").cast(pl.Binary), 'a cast to "Binary" is not read'),
    ],
)
def test_a_node_a_formula_cannot_carry_is_refused_by_name(expr, named):
    with pytest.raises(FormulaError, match="a window formula is element-wise") as e:
        po.ewm_mean(expr, half_life=5.0)
    assert named in str(e.value)


def test_the_operators_arguments_are_checked_by_name():
    with pytest.raises(
        TypeError, match="po.ewm_mean: the input must be a column name or a pl.Expr, got int"
    ):
        po.ewm_mean(3, half_life=5.0)  # type: ignore[arg-type]
    for bad, kind in ((True, "bool"), ([1.0], "list")):
        with pytest.raises(
            TypeError, match=f"po.ewm_mean: half_life must be a number or a duration, got {kind}"
        ):
            po.ewm_mean("x", half_life=bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="po.ewm_mean: half_life is required"):
        po.ewm_mean("x", half_life=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="po.ewm_sum: reverse must be a bool, got int"):
        po.ewm_sum("x", half_life=5.0, window_size=10.0, reverse=1)  # type: ignore[arg-type]


def test_from_tree_rebuilds_a_comparison_and_refuses_what_is_no_node():
    """``from_tree``, the inverse the tests hold ``to_tree`` to, rebuilds
    ``<`` and refuses a tree that is not a list headed by its kind, or one
    whose kind it does not read."""
    e = pl.col("x") < pl.col("y")
    assert from_tree(to_tree(e)).meta.eq(e)
    with pytest.raises(FormulaError, match="a formula node is a list headed by its kind, got 3"):
        from_tree(3)
    with pytest.raises(FormulaError, match="formula node 'sin' is not read"):
        from_tree(["sin", ["col", "x"]])
