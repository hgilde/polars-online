"""The dormant expression namespace (docs/PLAN.md section 6).

`pl.col("y").online.<model>(...)` exists only in a build with the `expr-plugin`
cargo feature (`maturin develop --features expr-plugin`). Tests that call it
carry this marker and skip otherwise; the static checks on the namespace class
(`test_kwargs_typing.py`, `test_model_registry.py`) run in every build.
"""

import pytest

import polars_online as po

BUILT = po.has_expr_plugin()
requires_expr_plugin = pytest.mark.skipif(
    not BUILT,
    reason="expression namespace not built (maturin develop --features expr-plugin;"
    " docs/PLAN.md section 6)",
)
