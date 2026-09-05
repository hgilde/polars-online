"""The `online` expression namespace (docs/PLAN.md section 6) -- in-memory only.

``pl.col("y").online.ewridge(features=[...], halflife=...)`` runs one spec over
the column the expression receives; use ``.over(group)`` for per-group streams.
Features are column names or named expressions (``pl.col("x").shift(1)
.alias("x_lag")``), evaluated per group under ``.over``. The implementation is
the model bank itself, so expression == bank by construction.

**Every call warns** (:class:`InMemoryExpressionWarning`). Polars hands a
stateful user expression its whole column in either engine -- its streaming
engine collects the input to do so -- so this form is O(data) where
``lf.online.fit_predict(specs)`` and ``po.run`` are O(chunk): 7.3 GB against
1.35 GB at 12M rows for the same model (docs/PERFORMANCE.md section 11). That
is polars' contract for a user expression, not something a plugin can change,
and a reader who takes the expression for the natural streaming form gets
the collecting one. The namespace stays for a frame already in memory, where
it is the shortest way to write the model and features can be expressions; the warning
exists so that nobody learns the difference from a memory profile. See the
README's closing section.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Unpack

import polars as pl
from polars.plugins import register_plugin_function

from polars_online import _spec
from polars_online._kwargs import (
    EwCovKwargs,
    EwridgeKwargs,
    FtrlKwargs,
    HoltKwargs,
    HuberKwargs,
    KalmanKwargs,
    KMeansKwargs,
    LassoKwargs,
    MicroKwargs,
    PaKwargs,
    QuantileKwargs,
    RlsKwargs,
    SgdKwargs,
)

_PLUGIN_PATH = Path(__file__).parent

__all__ = ["Feature", "InMemoryExpressionWarning", "OnlineNamespace", "online"]


class InMemoryExpressionWarning(UserWarning):
    """Issued by every ``pl.col(...).online.<model>(...)`` call: the expression
    form runs on the whole column at once.

    Polars calls a stateful user expression once with its whole column, in
    either engine, so in a plan over a file this form is O(data) where
    ``lf.online.fit_predict(specs)`` is O(chunk) -- the same model, the same
    numbers, and only one of them streams (module docstring). The warning is
    a ``UserWarning``, shown by default wherever the call is made; a
    ``DeprecationWarning`` would be hidden outside ``__main__``, which is the
    one place -- a pipeline module -- where it matters. Using the expression
    on a frame that is in memory anyway is fine; say so once::

        warnings.filterwarnings("ignore", category=po.InMemoryExpressionWarning)
    """


# Spec `type` -> namespace method, where the two differ.
_METHOD_OF = {"ew_ridge": "ewridge"}


def _warn_in_memory(kind: str, target: str) -> None:
    method = _METHOD_OF.get(kind, kind)
    msg = (
        f"polars_online: pl.col({target!r}).online.{method}(...) runs on the whole column at "
        "once -- polars hands a user expression its whole column in either engine -- so it "
        "is O(data) where lf.online.fit_predict([spec]) is O(chunk) for the same model. Fine "
        "for a frame in memory; for a stream write the model as a spec (README: 'The "
        "expression form'). Silence with warnings.filterwarnings('ignore', "
        "category=polars_online.InMemoryExpressionWarning)."
    )
    # stacklevel 4: this helper, `_run`, the namespace method, the user's call.
    warnings.warn(msg, InMemoryExpressionWarning, stacklevel=4)


Feature = str | pl.Expr
"""A feature is a column name or any expression with a determinable output
name (``pl.col("x").shift(1).alias("x_lag")``). Under ``.over(group)`` the
expression is evaluated per group, so a lag stays inside its group."""


def _features(features: list[Feature]) -> tuple[list[str], list[pl.Expr]]:
    """Split features into the names the spec carries and the expressions the
    plugin receives. An expression's output name is its feature name -- it is
    what ``coef`` fields and error messages call it -- so it has to be
    determinable and unique; ``.alias`` settles both."""
    names: list[str] = []
    exprs: list[pl.Expr] = []
    for f in features:
        if isinstance(f, str):
            names.append(f)
            exprs.append(pl.col(f))
        elif isinstance(f, pl.Expr):
            name = f.meta.output_name(raise_if_undetermined=False)
            if name is None:
                msg = (
                    "online: a feature expression must have a determinable output name "
                    "(give it an .alias)"
                )
                raise ValueError(msg)
            names.append(name)
            exprs.append(f)
        else:
            msg = f"online: features must be column names or expressions, got {type(f).__name__}"
            raise TypeError(msg)
    return names, exprs


def _run(spec: dict[str, Any], target_expr: pl.Expr, feature_exprs: list[pl.Expr]) -> pl.Expr:
    if spec["group"] is not None:
        # The Rust side would drop it silently: the expression always streams
        # over the column it receives, and polars does the grouping.
        msg = (
            f"online: group is not an expression parameter (the Rust side ignores it); "
            f"stream per group with .over({spec['group']!r}) instead"
        )
        raise TypeError(msg)
    _warn_in_memory(spec["model"]["type"], target_expr.meta.output_name())
    # ew_cov, kmeans and micro have no target: their first feature *is* the
    # calling column, so it must not be passed twice.
    if spec["model"]["type"] in _spec.UNSUPERVISED:
        args: list[pl.Expr] = []
    else:
        # The calling expression supplies the first target; any `extra_targets`
        # are ordinary columns. The order here must match `input_names` on the
        # Rust side: targets, then features, then clock/session/weight.
        args = [target_expr, *(pl.col(t) for t in spec["targets"][1:])]
    args += feature_exprs
    for col in (spec["clock"], spec["session"], spec["weight"]):
        if col is not None:
            args.append(pl.col(col))
    # Everything travels as ONE struct input. Polars evaluates a multi-input
    # group-aware function group by group on a single thread, but runs the
    # single-input path across its thread pool: 5x on 1000 groups (see
    # `online_run` in crates/online-py/src/expr.rs). Field names are
    # positional so that a column used twice (say as feature and weight)
    # cannot collide; the Rust side names them from the spec.
    packed = pl.struct([a.alias(f"_{i}") for i, a in enumerate(args)])
    out = register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="online_run",
        args=[packed],
        kwargs={"spec_json": _spec._json(spec)},
        is_elementwise=False,
        returns_scalar=False,
    )
    # Polars names a function's output after its first input, which is now the
    # packed struct; keep the name the calling column has always given it.
    return out.alias(target_expr.meta.output_name())


@pl.api.register_expr_namespace("online")
class OnlineNamespace:
    """Online models over the expression's column as a target.

    For a frame in memory. Polars calls the plugin once with the whole
    column, in either engine -- its streaming engine collects a user
    expression's input to do so -- so in a plan the column is O(data). For a
    stream, ``lf.online.fit_predict(specs)`` is the same bank as a plan that
    stays O(chunk) (:mod:`polars_online._frame`). Every method warns with
    :class:`InMemoryExpressionWarning` (module docstring).

    Each method takes the model's parameters as ``polars_online.spec``'s
    builder of the same name does, minus ``name``, ``targets`` and ``group``:
    the calling column is the target (``extra_targets`` adds more, sharing
    one fit), ``features`` are column names or named expressions, and a
    group is ``.over(group)``. Building the expression raises what the
    builder raises (:mod:`polars_online.spec`): ``TypeError`` for a keyword
    the model has not got or a value of the wrong shape, ``ValueError`` for
    a value the model refuses; and its own ``TypeError`` for ``group=``
    (written ``.over`` instead) or a feature that is neither a name nor an
    expression, ``ValueError`` for a calling or feature expression whose
    output name polars cannot determine (give it an ``.alias``), and for
    ``extra_targets`` naming the calling column or a column twice. When the
    expression runs, a column it reads that the frame has not got is polars'
    ``ColumnNotFoundError`` as the plan is resolved, and what the bank
    refuses on the data -- a column that is not numeric, a null clock, a
    negative weight, a clock running backwards under
    ``on_clock_reset="error"`` -- is polars' ``ComputeError`` (``the plugin
    failed with message: ...``) carrying the message
    :meth:`polars_online.ModelBank.fit_predict` gives for the same frame.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def _target(self) -> str:
        name = self._expr.meta.output_name(raise_if_undetermined=False)
        if name is None:
            msg = "online: the target expression must have a determinable name"
            raise ValueError(msg)
        return name

    def _targets(self, extra: list[str] | None) -> list[str]:
        """The calling column is the first target; `extra_targets` follow it.

        Multi-target specs share one `X'X`, so fitting several horizons in one
        call is much cheaper than one expression per target (ENHANCEMENTS E9).
        """
        first = self._target()
        rest = list(extra or [])
        if first in rest:
            msg = f"online: {first!r} is already the target this expression is called on"
            raise ValueError(msg)
        if len(set(rest)) != len(rest):
            msg = "online: extra_targets contains duplicates"
            raise ValueError(msg)
        return [first, *rest]

    def ewridge(
        self,
        features: list[Feature],
        extra_targets: list[str] | None = None,
        **kwargs: Unpack[EwridgeKwargs],
    ) -> pl.Expr:
        """EW-ridge over this column as the target. Same parameters as
        ``polars_online.spec.ewridge`` minus name/targets/group."""
        names, exprs = _features(features)
        spec = _spec.ewridge(
            "online",
            targets=self._targets(extra_targets),
            features=names,
            **kwargs,
        )
        return _run(spec, self._expr, exprs)

    def rls(
        self,
        features: list[Feature],
        extra_targets: list[str] | None = None,
        **kwargs: Unpack[RlsKwargs],
    ) -> pl.Expr:
        """Recursive least squares over this column as the target."""
        names, exprs = _features(features)
        spec = _spec.rls(
            "online",
            targets=self._targets(extra_targets),
            features=names,
            **kwargs,
        )
        return _run(spec, self._expr, exprs)

    def lasso(
        self,
        features: list[Feature],
        extra_targets: list[str] | None = None,
        **kwargs: Unpack[LassoKwargs],
    ) -> pl.Expr:
        """Lasso path with online lambda selection over this column as target."""
        names, exprs = _features(features)
        spec = _spec.lasso(
            "online",
            targets=self._targets(extra_targets),
            features=names,
            **kwargs,
        )
        return _run(spec, self._expr, exprs)

    def kalman(
        self,
        features: list[Feature],
        extra_targets: list[str] | None = None,
        **kwargs: Unpack[KalmanKwargs],
    ) -> pl.Expr:
        """Kalman / random-walk-beta filter over this column as the target."""
        names, exprs = _features(features)
        spec = _spec.kalman(
            "online",
            targets=self._targets(extra_targets),
            features=names,
            **kwargs,
        )
        return _run(spec, self._expr, exprs)

    def huber(
        self,
        features: list[Feature],
        extra_targets: list[str] | None = None,
        **kwargs: Unpack[HuberKwargs],
    ) -> pl.Expr:
        """Huber regression over this column as the target."""
        names, exprs = _features(features)
        spec = _spec.huber(
            "online",
            targets=self._targets(extra_targets),
            features=names,
            **kwargs,
        )
        return _run(spec, self._expr, exprs)

    def quantile(
        self,
        features: list[Feature],
        extra_targets: list[str] | None = None,
        **kwargs: Unpack[QuantileKwargs],
    ) -> pl.Expr:
        """Quantile regression over this column as the target."""
        names, exprs = _features(features)
        spec = _spec.quantile(
            "online",
            targets=self._targets(extra_targets),
            features=names,
            **kwargs,
        )
        return _run(spec, self._expr, exprs)

    def ftrl(
        self,
        features: list[Feature],
        extra_targets: list[str] | None = None,
        **kwargs: Unpack[FtrlKwargs],
    ) -> pl.Expr:
        """Online logistic regression (FTRL-proximal) over this column as the
        binary target. ``pred`` is a probability."""
        names, exprs = _features(features)
        spec = _spec.ftrl(
            "online",
            targets=self._targets(extra_targets),
            features=names,
            **kwargs,
        )
        return _run(spec, self._expr, exprs)

    def ew_cov(self, others: list[Feature], **kwargs: Unpack[EwCovKwargs]) -> pl.Expr:
        """EW moments of this column together with ``others``.

        Unlike the model namespaces this one has no target: the column the
        expression is called on becomes the first feature.
        """
        names, exprs = _features(others)
        spec = _spec.ew_cov("online", features=[self._target(), *names], **kwargs)
        return _run(spec, self._expr, [self._expr, *exprs])

    def sgd(
        self,
        features: list[Feature],
        extra_targets: list[str] | None = None,
        **kwargs: Unpack[SgdKwargs],
    ) -> pl.Expr:
        """SGD with pluggable losses over this column as the target."""
        names, exprs = _features(features)
        spec = _spec.sgd(
            "online",
            targets=self._targets(extra_targets),
            features=names,
            **kwargs,
        )
        return _run(spec, self._expr, exprs)

    def pa(
        self,
        features: list[Feature],
        extra_targets: list[str] | None = None,
        **kwargs: Unpack[PaKwargs],
    ) -> pl.Expr:
        """Passive-aggressive regression over this column as the target."""
        names, exprs = _features(features)
        spec = _spec.pa(
            "online",
            targets=self._targets(extra_targets),
            features=names,
            **kwargs,
        )
        return _run(spec, self._expr, exprs)

    def holt(self, extra_targets: list[str] | None = None, **kwargs: Unpack[HoltKwargs]) -> pl.Expr:
        """Holt's linear trend over this column -- level plus slope, no features.

        The only namespace method without a ``features`` argument, because the
        model has none: it extrapolates this column's own level and trend.
        """
        spec = _spec.holt("online", targets=self._targets(extra_targets), **kwargs)
        return _run(spec, self._expr, [])

    def kmeans(self, others: list[Feature], **kwargs: Unpack[KMeansKwargs]) -> pl.Expr:
        """EW k-means over this column together with ``others``.

        Like ``ew_cov`` this has no target: the column the expression is
        called on becomes the first feature. ``k`` is required. The struct
        holds ``cluster``, ``dist``, ``dist2``, ``n_eff`` and ``coef`` (the
        centres).
        """
        names, exprs = _features(others)
        spec = _spec.kmeans("online", features=[self._target(), *names], **kwargs)
        return _run(spec, self._expr, [self._expr, *exprs])

    def micro(self, others: list[Feature], **kwargs: Unpack[MicroKwargs]) -> pl.Expr:
        """Density-based (micro-cluster) clustering over this column together
        with ``others``.

        Like ``kmeans`` this has no target: the column the expression is
        called on becomes the first feature. ``eps`` is required. The struct
        holds ``cluster``, ``dist``, ``micro``, ``outlier``, ``n_clusters``,
        ``n_micro``, ``n_eff`` and ``coef`` (the established summaries).
        """
        names, exprs = _features(others)
        spec = _spec.micro("online", features=[self._target(), *names], **kwargs)
        return _run(spec, self._expr, [self._expr, *exprs])


def online(expr: pl.Expr) -> OnlineNamespace:
    """``expr.online`` as a plain function, so that a type checker can see it.

    A registered namespace is attached to ``pl.Expr`` at runtime, so to a type
    checker ``pl.col("y").online`` is an attribute that does not exist. This
    returns the same namespace, with its methods and their typed keywords
    (docs/IMPROVEMENTS.md U4) visible::

        df.with_columns(po.online(pl.col("y")).ewridge(features=["x0"], halflife=10.0))
    """
    return OnlineNamespace(expr)
