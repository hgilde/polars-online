"""The `online` expression namespace (docs/PLAN.md section 6).

``pl.col("y").online.ewridge(features=[...], halflife=...)`` runs one spec over
the column the expression receives; use ``.over(group)`` for per-group streams.
The implementation is the model bank itself, so expression == bank by
construction. Grids are allowed but produce wide structs; the bank is the
recommended surface for grids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from polars.plugins import register_plugin_function

from polars_online import _spec

_PLUGIN_PATH = Path(__file__).parent

__all__ = ["OnlineNamespace"]


def _run(spec: dict[str, Any], target_expr: pl.Expr) -> pl.Expr:
    # ew_cov has no target: its first feature *is* the calling column, so it
    # must not be passed twice.
    is_ew_cov = spec["model"]["type"] == "ew_cov"
    args: list[Any] = [] if is_ew_cov else [target_expr]
    args += [pl.col(f) for f in spec["features"]]
    for col in (spec["clock"], spec["session"], spec["weight"]):
        if col is not None:
            args.append(pl.col(col))
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="online_run",
        args=args,
        kwargs={"spec_json": _spec._json(spec)},
        is_elementwise=False,
        returns_scalar=False,
    )


@pl.api.register_expr_namespace("online")
class OnlineNamespace:
    """Online models over the expression's column as a target."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def _target(self) -> str:
        name = self._expr.meta.output_name(raise_if_undetermined=False)
        if name is None:
            msg = "online: the target expression must have a determinable name"
            raise ValueError(msg)
        return name

    def ewridge(self, features: list[str], **kwargs: Any) -> pl.Expr:
        """EW-ridge over this column as the target. Same parameters as
        ``polars_online.spec.ewridge`` minus name/targets/group."""
        spec = _spec.ewridge("online", targets=[self._target()], features=features, **kwargs)
        return _run(spec, self._expr)

    def rls(self, features: list[str], **kwargs: Any) -> pl.Expr:
        """Recursive least squares over this column as the target."""
        spec = _spec.rls("online", targets=[self._target()], features=features, **kwargs)
        return _run(spec, self._expr)

    def lasso(self, features: list[str], **kwargs: Any) -> pl.Expr:
        """Lasso path with online lambda selection over this column as target."""
        spec = _spec.lasso("online", targets=[self._target()], features=features, **kwargs)
        return _run(spec, self._expr)

    def kalman(self, features: list[str], **kwargs: Any) -> pl.Expr:
        """Kalman / random-walk-beta filter over this column as the target."""
        spec = _spec.kalman("online", targets=[self._target()], features=features, **kwargs)
        return _run(spec, self._expr)

    def huber(self, features: list[str], **kwargs: Any) -> pl.Expr:
        """Huber regression over this column as the target."""
        spec = _spec.huber("online", targets=[self._target()], features=features, **kwargs)
        return _run(spec, self._expr)

    def quantile(self, features: list[str], **kwargs: Any) -> pl.Expr:
        """Quantile regression over this column as the target."""
        spec = _spec.quantile("online", targets=[self._target()], features=features, **kwargs)
        return _run(spec, self._expr)

    def ftrl(self, features: list[str], **kwargs: Any) -> pl.Expr:
        """Online logistic regression (FTRL-proximal) over this column as the
        binary target. ``pred`` is a probability."""
        spec = _spec.ftrl("online", targets=[self._target()], features=features, **kwargs)
        return _run(spec, self._expr)

    def ew_cov(self, others: list[str], **kwargs: Any) -> pl.Expr:
        """EW moments of this column together with ``others``.

        Unlike the model namespaces this one has no target: the column the
        expression is called on becomes the first feature.
        """
        spec = _spec.ew_cov("online", features=[self._target(), *others], **kwargs)
        return _run(spec, self._expr)
