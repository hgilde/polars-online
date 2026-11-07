"""Formulas: a Polars expression over the window operators, kept as this
library's own compact tree.

An operator such as :func:`~polars_online.ops.ewm_mean` returns a genuine
``pl.Expr``: a column reference whose name carries the operator's compact
JSON after ``@po:``. So it composes with ``pl.col``, ``pl.lit``, numbers,
``when/then`` and the element-wise functions on either side of a Polars
operator. At build time :func:`to_tree` reads the expression's own tree
(``expr.meta.serialize(format="json")``, under the caller's Polars). It
walks seven node kinds -- ``Column``, ``Literal``, ``BinaryExpr``,
``Function`` (log, exp, abs, sqrt, pow, clip, fill_null, is_null,
is_not_null, negate), ``Ternary``, ``Cast`` and ``Alias`` -- into nested
lists, with the operators inlined as nodes of their own:

.. code-block:: python

    ["-", ["rewm_mean", ["col", "mid"], {"half_life": "10s", "window_size": "1m"}], ["col", "mid"]]

The serialized tree as a whole is unstable across Polars versions, which is
why only these nodes are read, here, and the Rust side rebuilds the formula
through Polars' public builders. A node outside the seven -- ``shift``,
``cum_sum``, a rolling window, ``over``, an aggregation -- would depend on
the chunking, where one chunk or a thousand must give the same numbers, and
is refused by name. :func:`from_tree`
rebuilds the expression in Python, for a round trip.
"""

from __future__ import annotations

import json
import math
import numbers
import re
from datetime import timedelta
from typing import Any

import polars as pl

from polars_online._duration import _INFINITY, duration_text
from polars_online._polars_online import parse_duration

#: The prefix of the column an operator stands in as.
PREFIX = "@po:"

#: The window operators, and the one that reads one row back.
OPERATORS = (
    "ewm_mean",
    "rewm_mean",
    "ewm_sum",
    "rewm_sum",
    "ewm_rate",
    "rewm_rate",
    "ewm_var",
    "ewm_std",
    "ewm_quantile",
    "rewm_quantile",
    "increment",
)

#: The operators that take ``bias``: a variance's own (task 212).
VARIANCES = ("ewm_var", "ewm_std")

#: The operators that take ``quantile``, the level (task 272 (d)).
QUANTILES = ("ewm_quantile", "rewm_quantile")

_BINARY = {
    "Plus": "+",
    "Minus": "-",
    "Multiply": "*",
    "TrueDivide": "/",
    "Eq": "==",
    "NotEq": "!=",
    "Lt": "<",
    "LtEq": "<=",
    "Gt": ">",
    "GtEq": ">=",
    "And": "&",
    "Or": "|",
    "LogicalAnd": "&",
    "LogicalOr": "|",
}

_UNARY = {"Negate": "neg", "Abs": "abs", "Exp": "exp"}

_DTYPES = {
    "Float64": pl.Float64,
    "Float32": pl.Float32,
    "Int64": pl.Int64,
    "Int32": pl.Int32,
    "Int16": pl.Int16,
    "Int8": pl.Int8,
    "UInt64": pl.UInt64,
    "UInt32": pl.UInt32,
    "UInt16": pl.UInt16,
    "UInt8": pl.UInt8,
    "Boolean": pl.Boolean,
    "String": pl.String,
    # A date or a time-of-day literal serializes as its integer under the
    # dtype's name, and is carried through a cast to it: read as the bare
    # integer, its text and its arithmetic were not Polars' (review
    # 2026-10-05, YB1).
    "Date": pl.Date,
    "Time": pl.Time,
}


#: What a formula reads around the operators, in the words a reader looks
#: for, each with the nodes of the compact tree it stands for. The refusal
#: message lists these words, and a test holds their nodes to the ones
#: :func:`from_tree` rebuilds and to the Rust reader's list, and their words
#: to the table in the windows guide, so the message cannot leave out a node
#: that is accepted.
ACCEPTED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("columns", ("col",)),
    ("literals", ("lit",)),
    ("arithmetic", ("+", "-", "*", "/")),
    ("negation", ("neg",)),
    ("comparisons", ("==", "!=", "<", "<=", ">", ">=")),
    ("& and |", ("&", "|")),
    ("log", ("log",)),
    ("exp", ("exp",)),
    ("abs", ("abs",)),
    ("sqrt", ("sqrt",)),
    ("pow", ("**",)),
    ("clip", ("clip",)),
    ("fill_null", ("fill_null",)),
    ("is_null", ("is_null",)),
    ("is_not_null", ("is_not_null",)),
    ("when/then/otherwise", ("when",)),
    ("cast", ("cast",)),
    ("alias", ("alias",)),
)


def accepted_text() -> str:
    """The words of :data:`ACCEPTED` as one English list."""
    words = [word for word, _ in ACCEPTED]
    return ", ".join(words[:-1]) + " and " + words[-1]


class FormulaError(ValueError):
    """An expression a formula cannot carry: a node that is not element-wise,
    a literal with no JSON form, a cast to a dtype not read."""


def _refuse(who: str, kind: str, detail: str = "") -> FormulaError:
    """The refusal of a node a formula cannot carry, named by ``kind`` (and
    ``detail``, which follows it in the sentence)."""
    return FormulaError(
        f"{who}: a formula over window operators reads each row alone, but its expression (a "
        f'column computation, such as pl.col("x") + 1) holds {kind}{detail}, which it cannot '
        f"read. The formula is computed a chunk at a time, so a part that reads other rows, "
        f"such as a shift, a cumulative or rolling function or an aggregation, would read "
        f"only its chunk's rows. Build the formula from "
        f"{accepted_text()} around the operators, and compute anything else as a column first."
    )


#: What a refused cast is told: the types a formula casts to.
_CASTS = (
    " (a formula casts only to Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, "
    "Float32, Float64, Boolean, String, Date or Time)"
)

#: Polars' serialized names of the binary operators a formula refuses, as
#: the reader writes them.
_REFUSED_OPERATORS = {
    "FloorDivide": "//",
    "Modulus": "%",
    "Xor": "^",
    "EqValidity": "eq_missing",
    "NotEqValidity": "ne_missing",
}

#: Polars' serialized names of the expressions that read other rows, as the
#: reader writes them.
_REFUSED_KINDS = {
    "Agg": "an aggregation",
    "Over": "the method over",
    "Window": "the method over",
    "Len": "pl.len()",
    "Not": "the operator ~",
    "Selector": "pl.all() or another selector",
    "Wildcard": "pl.all()",
    "RollingExpr": "a rolling function",
    "RollingExprBy": "a rolling function",
}

#: Function families Polars serializes under one name, the method in the
#: detail: ``.sin()`` is ``{"Trigonometry": "Sin"}``.
_FAMILIES = ("Boolean", "Pow", "Trigonometry", "StringExpr", "TemporalExpr")


def _method(name: str) -> str:
    """A serialized function's name as the method the reader called: Polars
    serializes ``.cum_sum()`` as ``CumSum``."""
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    # Polars' own methods spell the inverse functions as one word: arcsin.
    return re.sub(r"^arc_", "arc", snake)


def _reserved(who: str, column: str) -> FormulaError:
    """The refusal of a column whose name an operator's own column takes."""
    return FormulaError(
        f"{who}: the column {column!r} has a name starting with {PREFIX!r}, which is kept for "
        f"the window operators' own columns. Rename the column."
    )


def _literal(v: Any, who: str) -> tuple[Any, str | None]:
    """The JSON value of a serialized literal and, for a typed one
    (``pl.lit(x, dtype=...)``, or a numpy scalar), its dtype's name; or a
    refusal."""
    if v == "Null":
        return None, None
    if isinstance(v, dict) and len(v) == 1:
        ((kind, inner),) = v.items()
        if kind in ("Dyn", "Scalar") and isinstance(inner, dict) and len(inner) == 1:
            ((dtype, value),) = inner.items()
            if dtype == "Null":
                return None, None
            if value is None:
                # Polars serializes inf and nan alike, as a null value under
                # the float's type, so a formula cannot carry either; a clip
                # with one bound has a form of its own (task 159, F4).
                raise _refuse(
                    who,
                    "a literal that is inf or nan",
                    " (for a clip with one bound, leave the other out)",
                )
            if isinstance(value, (bool, int, float, str)):
                # A typed numeric literal keeps its type through a cast, so
                # its dtype and last bits are Polars' own (task 159, F3), and
                # so does a date's or a time's integer (YB1). A string or a
                # boolean is `Scalar` untyped too, and has one type to be.
                numeric = dtype in _DTYPES and dtype not in ("String", "Boolean")
                typed = dtype if kind == "Scalar" and numeric else None
                return value, typed
    raise _refuse(who, "a literal of a type a formula does not carry, such as a list or bytes")


def _walk(node: Any, who: str) -> Any:
    if isinstance(node, str) and node in _REFUSED_KINDS:
        raise _refuse(who, _REFUSED_KINDS[node])
    if not isinstance(node, dict) or len(node) != 1:
        raise _refuse(who, "a part of the expression it does not recognize")
    ((kind, body),) = node.items()
    if kind == "Column":
        if isinstance(body, str) and body.startswith(PREFIX):
            reserved = _reserved(who, body)
            try:
                op = json.loads(body[len(PREFIX) :])
            except json.JSONDecodeError:
                raise reserved from None
            # Only an operator's own node is one: a list headed by an
            # operator, with its input and, but for `increment`, its
            # parameters. A short list raised a bare IndexError where the tree
            # is walked, and a number was refused as holding no operator
            # (review round 4, PD3).
            if not (isinstance(op, list) and len(op) in (2, 3) and op[0] in OPERATORS):
                raise reserved
            return op
        return ["col", body]
    if kind == "Literal":
        value, typed = _literal(body, who)
        # A null literal is `["lit"]`: TOML has no null (review R2, P3).
        node = ["lit"] if value is None else ["lit", value]
        return ["cast", node, typed] if typed else node
    if kind == "BinaryExpr":
        op = body.get("op")
        if op not in _BINARY:
            raise _refuse(who, f"the operator {_REFUSED_OPERATORS.get(op, _method(str(op)))}")
        return [_BINARY[op], _walk(body["left"], who), _walk(body["right"], who)]
    if kind == "Ternary":
        return [
            "when",
            _walk(body["predicate"], who),
            _walk(body["truthy"], who),
            _walk(body["falsy"], who),
        ]
    if kind == "Cast":
        dtype = body.get("dtype")
        name = dtype.get("Literal") if isinstance(dtype, dict) else dtype
        if isinstance(name, dict) and len(name) == 1:
            # A dtype with parameters -- Datetime, Duration, Decimal, List,
            # Categorical, Enum, Struct -- is a mapping under its name, which
            # the lookup below could not hash (review 2026-10-05, YB2).
            raise _refuse(who, f"a cast to {json.dumps(next(iter(name)))}", _CASTS)
        if not isinstance(name, str) or name not in _DTYPES:
            raise _refuse(
                who,
                f"a cast to {json.dumps(name if isinstance(name, str) else dtype)[:60]}",
                _CASTS,
            )
        # Strict, Polars' default, unless the cast says otherwise; a wrapping
        # cast (`wrap_numerical=True`) has no form here.
        options = body.get("options", "Strict")
        if options == "Strict":
            return ["cast", _walk(body["expr"], who), name]
        if options == "NonStrict":
            return ["cast", _walk(body["expr"], who), name, "non_strict"]
        raise _refuse(who, f"a cast with options {json.dumps(options)}")
    if kind == "Alias":
        expr, name = body
        return ["alias", _walk(expr, who), name]
    if kind == "Function":
        fn = body.get("function")
        args = [_walk(a, who) for a in body.get("input", [])]
        if isinstance(fn, str):
            if fn in _REFUSED_KINDS:
                raise _refuse(who, _REFUSED_KINDS[fn])
            if fn in _UNARY and len(args) == 1:
                return [_UNARY[fn], args[0]]
            if fn == "Log" and len(args) == 2:
                return ["log", args[0], args[1]]
            if fn == "FillNull" and len(args) == 2:
                return ["fill_null", args[0], args[1]]
            raise _refuse(who, f"the method {_method(fn)}")
        if isinstance(fn, dict) and len(fn) == 1:
            ((name, detail),) = fn.items()
            if name == "Pow" and detail == "Generic" and len(args) == 2:
                return ["**", args[0], args[1]]
            if name == "Pow" and detail == "Sqrt" and len(args) == 1:
                return ["sqrt", args[0]]
            if name == "Boolean" and detail in ("IsNull", "IsNotNull") and len(args) == 1:
                return ["is_null" if detail == "IsNull" else "is_not_null", args[0]]
            if name == "Clip" and isinstance(detail, dict):
                rest = args[1:]
                lo = rest.pop(0) if detail.get("has_min") else ["lit"]
                hi = rest.pop(0) if detail.get("has_max") else ["lit"]
                return ["clip", args[0], lo, hi]
            for part in (name, detail):
                if isinstance(part, str) and part in _REFUSED_KINDS:
                    raise _refuse(who, _REFUSED_KINDS[part])
            shown = name
            if name in _FAMILIES and isinstance(detail, str):
                shown = detail
            elif name in _FAMILIES and isinstance(detail, dict) and detail:
                shown = str(next(iter(detail)))
            raise _refuse(who, f"the method {_method(shown)}")
        if isinstance(fn, dict) and fn:
            raise _refuse(who, f"the method {_method(str(next(iter(fn))))}")
        raise _refuse(who, "a function it does not recognize")
    raise _refuse(who, _REFUSED_KINDS.get(kind, "a part of the expression it does not recognize"))


def to_tree(expr: pl.Expr, who: str = "window formula") -> list[Any]:
    """The compact tree of ``expr``, the operators inlined.

    Raises :class:`FormulaError` for an expression a formula cannot carry,
    naming ``who``, the call the expression was given to.
    """
    return _walk(json.loads(expr.meta.serialize(format="json")), who)


def operators(tree: Any) -> list[list[Any]]:
    """Every operator node in ``tree``, outermost first."""
    out: list[list[Any]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if head in OPERATORS:
            out.append(node)
            walk(node[1])
        elif head not in ("col", "lit"):
            for arg in node[1:]:
                walk(arg)

    walk(tree)
    return out


#: The operators that look ahead: a formula with one is a target of the
#: row's future (docs/PLAN.md task 104).
FORWARD = ("rewm_mean", "rewm_sum", "rewm_rate", "rewm_quantile")


def columns(tree: Any) -> list[str]:
    """Every column ``tree`` reads, in first-seen order."""
    out: list[str] = []

    def walk(node: Any) -> None:
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if head == "col":
            if node[1] not in out:
                out.append(node[1])
        elif head == "lit":
            return
        elif head in OPERATORS:
            walk(node[1])
        else:
            for arg in node[1:]:
                walk(arg)

    walk(tree)
    return out


def looks_ahead(tree: Any) -> bool:
    """Whether ``tree`` holds an operator over the rows after each row."""
    return any(op[0] in FORWARD for op in operators(tree))


def compact(node: Any) -> str:
    """One operator node as the text its column carries: compact JSON with
    sorted keys, so one operator asked for twice is one column."""
    return json.dumps(node, separators=(",", ":"), sort_keys=True)


def operator_column(node: list[Any]) -> pl.Expr:
    """The column an operator stands in as."""
    return pl.col(PREFIX + compact(node))


def from_tree(tree: Any) -> pl.Expr:
    """The expression a tree stands for, rebuilt with Polars' builders; an
    operator comes back as its column."""
    if not isinstance(tree, list) or not tree or not isinstance(tree[0], str):
        raise FormulaError(
            f"from_tree: a formula node is a list headed by its kind, but it is {tree!r}. "
            f"Give it a tree that to_tree wrote."
        )
    head, args = tree[0], tree[1:]
    if head in OPERATORS:
        return operator_column(tree)
    if head == "col":
        return pl.col(args[0])
    if head == "lit":
        return pl.lit(args[0]) if args else pl.lit(None)
    if head == "clip":
        # A missing bound is the null literal `["lit"]` (or a bare null in
        # a tree written before TOML could not carry one).
        lo = None if args[1] in (None, ["lit"]) else from_tree(args[1])
        hi = None if args[2] in (None, ["lit"]) else from_tree(args[2])
        return from_tree(args[0]).clip(lo, hi)
    e = [from_tree(a) for a in args if isinstance(a, list)]
    match head:
        case "+":
            return e[0] + e[1]
        case "-":
            return e[0] - e[1]
        case "*":
            return e[0] * e[1]
        case "/":
            return e[0] / e[1]
        case "**":
            return e[0] ** e[1]
        case "==":
            return e[0] == e[1]
        case "!=":
            return e[0] != e[1]
        case "<":
            return e[0] < e[1]
        case "<=":
            return e[0] <= e[1]
        case ">":
            return e[0] > e[1]
        case ">=":
            return e[0] >= e[1]
        case "&":
            return e[0] & e[1]
        case "|":
            return e[0] | e[1]
        case "neg":
            return -e[0]
        case "abs":
            return e[0].abs()
        case "exp":
            return e[0].exp()
        case "sqrt":
            return e[0].sqrt()
        case "is_null":
            return e[0].is_null()
        case "is_not_null":
            return e[0].is_not_null()
        case "log":
            return e[0].log(args[1][1]) if args[1][0] == "lit" else e[0].log() / e[1].log()
        case "fill_null":
            return e[0].fill_null(e[1])
        case "when":
            return pl.when(e[0]).then(e[1]).otherwise(e[2])
        case "cast":
            return from_tree(args[0]).cast(_DTYPES[args[1]], strict=len(args) < 3)
        case "alias":
            return from_tree(args[0]).alias(args[1])
    raise FormulaError(
        f"from_tree: the formula node {head!r} is not one it reads. Give it a tree that to_tree "
        f"wrote."
    )


def _input_tree(who: str, value: Any) -> list[Any]:
    if isinstance(value, str):
        if value.startswith(PREFIX):
            raise _reserved(who, value)
        return ["col", value]
    if isinstance(value, pl.Expr):
        tree = to_tree(value, who)
        if any(op[0] != "increment" for op in operators(tree)):
            raise ValueError(
                f"{who}: the input holds another window operator, but an operator's input may "
                f"hold only columns and Polars expressions of the same row (a column "
                f'computation, such as pl.col("x") + 1), or po.increment. To apply an operator to '
                f"another's output, add the inner one as a column with po.stream.with_windows "
                f"first, and pass that column."
            )
        return tree
    raise TypeError(
        f"{who}: the input must be a column name or a pl.Expr, but it is of type "
        f'{type(value).__name__}. Pass a column name or an expression, such as pl.col("x").'
    )


def _span(who: str, key: str, value: Any, *, inf_ok: bool) -> float | str:
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, str, timedelta, pl.Expr)):
        raise TypeError(
            f"{who}: {key} must be a number or a duration, but it is of type "
            f"{type(value).__name__}. "
            f'Give {key} a number of clock units, or a duration such as "5m".'
        )
    # A word is checked as the number it names, and a duration as its length,
    # here, as a number is: "nan", "-inf", "reset", an infinite window and a
    # duration at or below 0 passed, and were refused only when the plan was
    # built, under a serde path (review round 4, PD9).
    if isinstance(value, str):
        word = value.strip().lower()
        if word == "reset":
            # `session_gap`'s word, which the clock parameters' words admit.
            raise ValueError(
                f"{who}: {key} must be a number or a duration, but it is {value!r}. "
                f'Give {key} a number of clock units, or a duration such as "5m".'
            )
        if word == "nan" or word in _INFINITY:
            _number(who, key, math.nan if word == "nan" else _INFINITY[word], value, inf_ok=inf_ok)
            return value.strip()
    if isinstance(value, numbers.Real):
        return _number(who, key, float(value), value, inf_ok=inf_ok)
    text = duration_text(value, who, key)
    if parse_duration(text) <= 0:
        raise ValueError(
            f"{who}: {key} must be above 0, but it is {text}. Give {key} a positive duration."
        )
    return text


def _number(who: str, key: str, x: float, shown: Any, *, inf_ok: bool) -> float | str:
    """A clock parameter given as the number ``x`` (``shown`` as written):
    above 0, and finite unless ``inf_ok``; an infinity is ``"inf"``."""
    above = "above 0" if inf_ok else "finite and above 0"
    inf = ", or inf for no decay" if inf_ok else ""
    if math.isnan(x):
        raise ValueError(
            f"{who}: {key} must be {above}, but it is NaN. Give {key} a positive number of "
            f"clock units, or a duration{inf}."
        )
    if math.isinf(x):
        if not inf_ok or x < 0:
            raise ValueError(
                f"{who}: {key} must be {above}, but it is {shown}. Give {key} a positive "
                f"number of clock units, or a duration{inf}."
            )
        return "inf"
    if x <= 0:
        raise ValueError(
            f"{who}: {key} must be {above}, but it is {shown}. Give {key} a positive number of "
            f"clock units, or a duration{inf}."
        )
    return x


#: The most rows ``min_samples`` can ask for: the windows core counts them
#: in a ``u32``.
MAX_MIN_SAMPLES = 2**32 - 1


def operator(
    name: str,
    input: Any,
    *,
    half_life: Any = None,
    window_size: Any = None,
    closed: str = "right",
    min_samples: int = 1,
    partial: str | None = None,
    bias: Any = None,
    quantile: Any = None,
    who: str | None = None,
) -> pl.Expr:
    """The expression of operator ``name`` over ``input`` with these
    parameters, checked here so a wrong value names its parameter; ``bias``
    a variance's alone, ``quantile`` a quantile's. ``who`` is the call a
    message names, ``po.<name>`` unless given: a window looking ahead is
    ``ewm_*(reverse=True)`` to the caller and ``rewm_*`` in the tree."""
    who = who or f"po.{name}"
    tree = _input_tree(who, input)
    if name == "increment":
        return operator_column(["increment", tree])
    if half_life is None:
        raise TypeError(
            f"{who}: half_life is required, but it is not given. Give half_life in clock units, "
            f"or inf for no decay."
        )
    params: dict[str, Any] = {"half_life": _span(who, "half_life", half_life, inf_ok=True)}
    if window_size is not None:
        params["window_size"] = _span(who, "window_size", window_size, inf_ok=False)
    elif name.startswith("rewm"):
        raise TypeError(
            f"{who}: window_size is required when the operator looks ahead, since the rows "
            f"after a row have no end without one, but it is not given. Give window_size in "
            f"clock units."
        )
    if closed not in ("right", "left", "both", "none"):
        raise ValueError(
            f'{who}: closed must be "right", "left", "both" or "none", but it is {closed!r}. '
            f"Set closed to one of those words."
        )
    params["closed"] = closed
    # Any integer Polars' `rolling_*_by` takes, a numpy one included, up to
    # the core's ceiling, named; past it the refusal came later and said "at
    # least 1" (review round 4, PD4).
    if (
        isinstance(min_samples, bool)
        or not isinstance(min_samples, numbers.Integral)
        or not 1 <= min_samples <= MAX_MIN_SAMPLES
    ):
        raise ValueError(
            f"{who}: min_samples must be an integer of at least 1 and at most "
            f"{MAX_MIN_SAMPLES}, but it is {min_samples!r}. Set min_samples to a whole number "
            f"of rows in that range."
        )
    params["min_samples"] = int(min_samples)
    if partial is not None:
        if partial not in ("keep", "null", "drop"):
            raise ValueError(
                f'{who}: partial must be "keep", "null" or "drop", but it is {partial!r}. '
                f"Set partial to one of those words."
            )
        if window_size is None:
            # What a window cut short gives needs a window to cut; without
            # one it was taken and did nothing (review 2026-10-05, YB5).
            raise ValueError(
                f"{who}: partial needs window_size, but window_size is not given. partial says "
                f"what a window cut short by a clock break (a step past gap_cap, or a session "
                f"change) gives, and without window_size no window is cut short. Give "
                f"window_size, or leave out partial."
            )
        params["partial"] = partial
    if name in VARIANCES:
        # A bool, as Polars' `bias` is; written always, as `closed` is, so
        # one variance asked for twice is one column.
        if not isinstance(bias, bool):
            raise TypeError(
                f"{who}: bias must be True or False, but it is of type {type(bias).__name__}. "
                f"Set bias to True or False."
            )
        params["bias"] = bias
    elif bias is not None:
        raise TypeError(
            f"{who}: bias is given, but only a variance (ewm_var, ewm_std) takes it. "
            f"Leave out bias."
        )
    if name in QUANTILES:
        # A number in (0, 1), as Polars' `quantile` takes it; NaN and the
        # ends refused by name.
        if isinstance(quantile, bool) or not isinstance(quantile, numbers.Real):
            raise TypeError(
                f"{who}: quantile must be a number, but it is of type {type(quantile).__name__}. "
                f"Set quantile to a number above 0 and below 1."
            )
        if not 0.0 < float(quantile) < 1.0:
            raise ValueError(
                f"{who}: quantile must be above 0 and below 1, but it is {quantile!r}. "
                f"Set quantile to a number above 0 and below 1, such as 0.5 for the median."
            )
        params["quantile"] = float(quantile)
    elif quantile is not None:
        raise TypeError(
            f"{who}: quantile is given, but only ewm_quantile takes it. Leave out quantile."
        )
    return operator_column([name, tree, params])
