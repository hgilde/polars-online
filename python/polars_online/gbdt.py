"""XGBoost's and LightGBM's training inputs, built from Arrow IPC files,
Parquet files or a query, in bounded memory.

Both libraries train in two stages. *Binning* places each feature's bins and
puts every value into one; *boosting* grows trees from per-bin sums and never
reads a raw value again. This module does the first stage from data that
streams, in Rust, and hands back the library's own training object, so that
``xgb.train`` and ``lgb.train`` run unchanged, with their evaluation sets,
early stopping and callbacks. Each function requires its library
(``pip install xgboost`` or ``lightgbm``), which is not a dependency of this
package, and numpy, which both libraries require. This code reads the training
rows by a glob from the two parquet files that `Example data
<https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_ writes:

.. code-block:: python

    import xgboost as xgb

    features = ["x0", "x1", "x2", "signal_a", "signal_b"]
    train = po.gbdt.xgboost_matrix("ticks/*.parquet", label="y", features=features)
    valid = po.gbdt.xgboost_matrix("today.parquet", label="y", features=features, ref=train)
    booster = xgb.train({"tree_method": "hist", "max_bin": 256}, train, 500,
                        evals=[(valid, "valid")], early_stopping_rounds=20, verbose_eval=False)

- :func:`xgboost_matrix` returns an ``xgboost.QuantileDMatrix``. XGBoost reads
  its input four times, and the second pass sketches every feature's quantiles
  over every row. This places the cuts from a sample of ``sample_rows`` rows
  instead, which replaces XGBoost's sketch of every row with a pass that draws
  the sample, and halves the time before the first round (4.0 s against 8.0 s
  from Arrow IPC files, as :func:`xgboost_matrix`'s Notes measure). Without
  weights the sample is uniform. On 10 million unweighted rows of 50 features,
  at 50 rounds, XGBoost's sketch scored a validation RMSE of 1.2379, and cuts
  from five seeds of the sample 1.2347 to 1.2411. That spread is no wider than
  LightGBM's own sampling gives: 1.2298 to 1.2443 over five seeds. With
  ``weight=``, the sample is drawn in proportion to the weights and its
  quantiles are weighted, as XGBoost's sketch is. On 50,000 normal values with
  weight 50 above 1.0 and ``max_bin=16``, 14 of the 15 cuts fall above 1.0, as
  14 of XGBoost's own do, and the heaviest bin holds 6.26% of the weight,
  against 6.47% for XGBoost's. ``sample_rows=None`` keeps XGBoost's sketch and
  its own trees.
- :func:`lightgbm_dataset` returns a constructed ``lightgbm.Dataset``, binned
  on LightGBM's own sample: a first pass gathers the rows LightGBM samples, a
  second pushes every row as float32; or, when the data fits, the whole stream
  goes in as Arrow (under the native engine).

**The native engine** reads Arrow IPC files itself, one block of the file at
a time, and any other source as Polars gives the query's result, one chunk at
a time.
For XGBoost it hands each chunk's Arrow buffers to the library's C API with
no copy: a null goes as Arrow's validity bit, not as a NaN copied in. For
LightGBM it copies each chunk into float32 rows, with NaN for a null, or
under ``in_memory`` casts the stream to float32 Arrow. The library is
loaded from the user's own installed package, by the path that package loaded
it from. Nothing is linked into this package, whose only dependency stays
``polars``. ``engine="python"`` builds through the libraries' Python APIs
instead: numpy chunks into an ``xgboost.DataIter`` with the same sample's
cuts, and a ``lightgbm.Sequence``. ``engine="auto"`` falls back to it, with a
warning, when the native engine cannot load the library. Both engines grow the
same trees.

**What the columns may hold.** Every feature is numeric (any integer or float
dtype), Boolean, or a ``pl.Enum``, whose codes go in as a categorical
feature. The Enum's categories fix what each code means, so a model is read
the same way at prediction (pass the codes, ``pl.col(c).to_physical()``). A
``pl.Categorical`` is refused, since its codes depend on the process that
made them; cast it to a ``pl.Enum``. The label and the weight are numeric,
with no nulls and finite values (a weight not negative). A margin
(``base_margin=``, ``init_score=``) is a number a row, or a ``List`` or
``Array`` of one number for each class, finite and never null.

**How the figures in these docstrings were measured.** On an Apple M4 Pro
(14 cores, 48 GB), each route ran in a fresh process with 8 threads for the
library, for Polars and for Rust. Other jobs shared the machine, so the
routes ran interleaved, three times each, and a figure is the median, at a
load average of 4.5 to 10. Memory is the peak physical footprint. The rows
were seeded: 50 Float64 features of four kinds (normal, lognormal, uniform,
and a normal rounded to 0.1), 5% nulls in every fifth column, one
``pl.Enum`` of 32 levels drawn unevenly, and a target built from them plus
noise. Accuracy is the validation RMSE on separate rows from another seed.

.. warning::
    This module is considered **unstable**: it is built on the libraries' C
    APIs and on how their Python classes hold a native handle. So its
    keywords and what it supports may change in any release without that
    counting as a breaking change, and a new library version may need a new
    release here. Under ``POLARS_ONLINE_WARN_UNSTABLE=1``, each call issues
    :class:`polars_online.UnstableWarning`.
"""

from __future__ import annotations

import ctypes
import functools
import glob
import math
import numbers
import operator
import os
import warnings
import weakref
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from polars_online import _polars_online as _native
from polars_online._errors import ColumnNotFoundError, WrongTypeError
from polars_online._warnings import unstable

if TYPE_CHECKING:
    import lightgbm
    import xgboost

__all__ = ["lightgbm_dataset", "xgboost_matrix"]

#: What a training source may be: Arrow IPC or Parquet paths (globs taken),
#: a ``LazyFrame``, or a ``DataFrame``.
Source = str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | pl.LazyFrame | pl.DataFrame

Engine = Literal["auto", "native", "python"]

#: Rows per batch the source is read in.
_CHUNK = 1 << 18
#: Batches the native engine's reader keeps ready ahead of the library: on 10
#: million rows of 50 features, 1, 2 and 4 took the same time, and each batch
#: held costs one batch of memory (100 MB there).
_READ_AHEAD = 2
_ENGINES = ("auto", "native", "python")
#: The last native build's counts and timings (rows, passes, seconds
#: sampling, building and waiting for a batch), for
#: scripts/gbdt_adapter.py's measurements; not part of the API.
_LAST_STATS: dict[str, float] = {}


def _listed(names: Sequence[Any]) -> str:
    """Names for a message: the list whole up to ten, else its first ten and
    the count of the rest."""
    names = list(names)
    if len(names) <= 10:
        return repr(names)
    return f"{names[:10]!r} and {len(names) - 10} more"


def _format(path: str | os.PathLike[str], who: str) -> str:
    ext = Path(os.fspath(path)).suffix.lower()
    if ext in (".arrow", ".ipc", ".feather"):
        return "ipc"
    if ext in (".parquet", ".pq"):
        return "parquet"
    raise ValueError(
        f"{who}: the format of {os.fspath(path)!r} cannot be told from its extension, and "
        "po.gbdt reads Arrow IPC (.arrow, .ipc, .feather) and Parquet (.parquet, .pq) files. "
        "Rename the file to one of those extensions, or pass a query that reads it."
    )


def _expand(paths: list[str | os.PathLike[str]], who: str) -> list[str]:
    """The files a list of paths and globs names, each glob's sorted, as
    polars' scans order them."""
    out: list[str] = []
    for p in paths:
        s = os.fspath(p)
        if any(c in s for c in "*?["):
            found = sorted(glob.glob(s))
            if not found:
                raise FileNotFoundError(
                    f"{who}: no file matches the pattern {s!r}. Check the path and the "
                    "pattern's wildcards."
                )
            out.extend(found)
        else:
            out.append(s)
    return out


def _plan(source: Source, who: str) -> tuple[pl.LazyFrame, list[str] | None]:
    """The source as a plan, and its files when they are Arrow IPC, which
    the native engine reads itself."""
    if isinstance(source, pl.LazyFrame):
        return source, None
    if isinstance(source, pl.DataFrame):
        return source.lazy(), None
    if isinstance(source, (str, os.PathLike)):
        paths: list[str | os.PathLike[str]] = [source]
    elif isinstance(source, Sequence):
        paths = list(source)
    else:
        raise TypeError(
            f"{who}: source must be a path, a list of paths, a query or a frame, but its type "
            f"is {type(source).__name__}. Pass one of those."
        )
    if not paths:
        raise ValueError(
            f"{who}: source must name at least one file, but it is empty. Pass the files to read."
        )
    formats = {_format(p, who) for p in paths}
    if len(formats) > 1:
        raise ValueError(
            f"{who}: source must be files of one format, but it mixes Arrow IPC and Parquet "
            "files. Pass files of one format."
        )
    names = [os.fspath(p) for p in paths]
    if formats.pop() == "ipc":
        return pl.scan_ipc(names), _expand(paths, who)
    # A local glob that matches nothing is refused now, as for Arrow IPC
    # above: Polars' Parquet scan reports it only when the files are read,
    # and as a `ComputeError`. A URL's glob is Polars' to expand.
    _expand([n for n in names if "://" not in n], who)
    return pl.scan_parquet(names), None


class _Columns:
    """The features, label, weight and margin of a source, checked, and the
    plan that selects and casts them as both engines read them: a
    ``pl.Enum`` as its codes, a Boolean as ``UInt8``, the label and weight
    as ``Float64``, the margin as ``Float64`` or a list of them."""

    def __init__(
        self,
        plan: pl.LazyFrame,
        *,
        label: str,
        weight: str | None,
        features: Sequence[str] | None,
        who: str,
        margin: str | None = None,
        margin_role: str = "base_margin",
    ) -> None:
        schema = plan.collect_schema()
        for role, name in (("label", label), ("weight", weight), (margin_role, margin)):
            if name is not None and name not in schema:
                raise ColumnNotFoundError(
                    f"{who}: the {role} column {name!r} is not in the source, which has "
                    f"{_listed(list(schema))}. Name a column the source has."
                )
        if label == margin or (weight is not None and weight == margin):
            raise ValueError(
                f"{who}: {margin!r} is named as the {margin_role} and as the label or weight. "
                f"Name another column as the {margin_role}."
            )
        own = (label, weight, margin)
        if features is None:
            names = [c for c in schema if c not in own]
        else:
            if isinstance(features, str):
                raise TypeError(
                    f"{who}: features must be a list of column names, but it is the one name "
                    f"{features!r}. Pass it in a list, as [{features!r}]."
                )
            names = list(features)
            missing = [c for c in names if c not in schema]
            if missing:
                raise ColumnNotFoundError(
                    f"{who}: features names columns the source does not have, {missing}, and "
                    f"the source has {_listed(list(schema))}. Name columns the source has."
                )
            clash = [c for c in names if c in (label, weight)]
            if clash:
                raise ValueError(
                    f"{who}: {clash[0]!r} is named as a feature and as the label or weight, but "
                    "a column can be only one of them. Remove it from features."
                )
            if margin is not None and margin in names:
                raise ValueError(
                    f"{who}: {margin!r} is named as a feature and as the {margin_role}. Add a "
                    "copy in the query you pass as source, as "
                    f"with_columns(pl.col({margin!r}).alias({margin + '_feature'!r})), and name "
                    "the copy in features."
                )
            if len(set(names)) != len(names):
                twice = next(c for c in names if names.count(c) > 1)
                raise ValueError(
                    f"{who}: features names column {twice!r} twice. Name each column once."
                )
        if not names and features is not None:
            raise ValueError(
                f"{who}: features must name at least one column, but it is empty. Name the "
                "feature columns, or leave features out to take every other column."
            )
        if not names:
            taken = ["the label"] + ["the weight"] * (weight is not None)
            taken += [f"the {margin_role}"] * (margin is not None)
            besides = (
                " and ".join(taken)
                if len(taken) < 3
                else "the label, the weight and the " + margin_role
            )
            raise ValueError(
                f"{who}: the source has no column besides {besides} to be a feature. Add a "
                "feature column to the source."
            )
        self.names = names
        #: The call the reader made, which every refusal of a row names.
        self.who = who
        self.label = label
        self.weight = weight
        self.margin = margin
        self.margin_role = margin_role
        #: The margin's values a row, and the row that set it (``None``
        #: when the training set's margin set it), for the Python engine.
        self.margin_width: tuple[int, int | None] | None = None
        self.kinds: list[str] = []
        self.levels: list[int] = []
        #: Each feature's Enum categories, for the native IPC reader.
        self.categories: list[list[str] | None] = []
        exprs: list[pl.Expr] = []
        for c in names:
            dt = schema[c]
            if dt == pl.Boolean:
                self.kinds.append("bool")
                exprs.append(pl.col(c).cast(pl.UInt8))
                self.levels.append(0)
            elif dt.is_integer():
                self.kinds.append("int")
                exprs.append(pl.col(c))
                self.levels.append(0)
            elif dt.is_float():
                self.kinds.append("float")
                exprs.append(
                    pl.col(c) if dt in (pl.Float32, pl.Float64) else pl.col(c).cast(pl.Float32)
                )
                self.levels.append(0)
            elif isinstance(dt, pl.Enum):
                self.kinds.append("category")
                exprs.append(pl.col(c).to_physical())
                self.levels.append(len(dt.categories))
                self.categories.append(dt.categories.to_list())
            elif isinstance(dt, pl.Categorical):
                raise WrongTypeError(
                    f"{who}: feature {c!r} is a pl.Categorical, whose codes depend on the "
                    "process that made them, so a model trained on them could not read them "
                    "again. Cast it to a pl.Enum, whose categories fix the codes."
                )
            else:
                raise WrongTypeError(
                    f"{who}: feature {c!r} is {dt}, but po.gbdt reads integer, float, Boolean "
                    "and pl.Enum columns. Cast it to Float64 or a pl.Enum, or leave it out of "
                    "features."
                )
            if len(self.categories) < len(self.kinds):
                self.categories.append(None)
        for role, name in (("label", label), ("weight", weight)):
            if name is None:
                continue
            dt = schema[name]
            if not (dt.is_numeric() or dt == pl.Boolean) or dt.is_decimal():
                raise WrongTypeError(
                    f"{who}: the {role} column {name!r} is {dt}, but it must be an integer, "
                    "float or Boolean column. Cast it to Float64."
                )
            exprs.append(pl.col(name).cast(pl.Float64))
        if margin is not None:
            exprs.append(_margin_expr(who, margin_role, margin, schema[margin]))
        self.plan = plan.select(exprs)

    @property
    def xgb_types(self) -> list[str]:
        """XGBoost's feature types, as its Arrow route records them."""
        return [
            {"float": "float", "int": "int", "bool": "i", "category": "c"}[k] for k in self.kinds
        ]

    def batches(self) -> Iterator[pl.DataFrame]:
        return iter(self.plan.collect_batches(chunk_size=_CHUNK, maintain_order=True))

    def check_meta(self, frame: pl.DataFrame, row: int) -> None:
        """The Python engine's refusal of a null or non-finite label or
        weight, as the native engine's."""
        for role, name in (("label", self.label), ("weight", self.weight)):
            if name is None:
                continue
            s = frame[name]
            if s.null_count():
                at = row + int(s.is_null().arg_max() or 0)
                raise ValueError(
                    f"{self.who}: the {role} column `{name}` has a null at row {at} (counting from "
                    f"0), and every row needs a {role}. Replace the null, or drop the row, before "
                    "the build."
                )
            bad = ~s.is_finite() if role == "label" else (~s.is_finite()) | (s < 0)
            if bad.any():
                i = int(bad.arg_max() or 0)
                need = "finite" if role == "label" else "finite and not negative"
                v = float(s[i])
                shown = "NaN" if math.isnan(v) else repr(v)
                raise ValueError(
                    f"{self.who}: the {role} column `{name}` holds {shown} at row {row + i} "
                    f"(counting from 0), and a {role} must be {need}. Replace it, or drop the row, "
                    "before the build."
                )

    def margin_numpy(self, frame: pl.DataFrame, row: int) -> Any:
        """The Python engine's margin of a batch: ``n`` values, or ``n``
        rows of ``k``, refused by name as the native engine refuses it."""
        assert self.margin is not None
        s = frame[self.margin]
        n = s.len()
        if n == 0:
            return None
        scalar = not isinstance(s.dtype, (pl.List, pl.Array))
        if scalar:
            lens = pl.repeat(1, n, dtype=pl.UInt32, eager=True)
            bad = ~s.is_finite().fill_null(False)
        else:
            s = s.cast(pl.List(pl.Float64))
            lens = s.list.len()
            bad = s.is_null() | s.list.eval(~pl.element().is_finite().fill_null(False)).list.any()
        if self.margin_width is None:
            first = lens[0]
            if first is None or first == 0:
                self._refuse_margin(s[0], row)
            self.margin_width = (int(first), row)
        k = self.margin_width[0]
        bad = bad.fill_null(True) | (lens != k).fill_null(True)
        if bad.any():
            i = int(bad.arg_max() or 0)
            self._refuse_margin(s[i], row + i)
        if scalar:
            return s.to_numpy()
        flat = s.explode().to_numpy()
        return flat if k == 1 else flat.reshape(n, k)

    def _refuse_margin(self, value: Any, at: int) -> None:
        """The native engine's refusal of row ``at``'s margin, word for word."""
        role, name = self.margin_role, self.margin
        whose = "ref" if role == "base_margin" else "reference"
        who = self.who
        null = ValueError(
            f"{who}: the {role} column `{name}` has a null at row {at} (counting from 0), and "
            "every row needs a margin. Replace the null before the build, with 0 where the row "
            "has no prediction to start from."
        )
        if value is None:
            raise null
        values = list(value) if isinstance(value, pl.Series) else [value]
        k = len(values)
        if self.margin_width is None:
            raise ValueError(
                f"{who}: the {role} column `{name}` holds no values at row {at} (counting from 0), "
                "and a margin needs one. Give every row one value, or one for each class."
            )
        want, set_at = self.margin_width
        if k != want:
            if set_at is None:
                raise ValueError(
                    f"{who}: the {role} column `{name}` holds {_values(k)} at row {at} (counting "
                    f"from 0), and the {whose}'s {role} holds {_values(want)} a row. Give every "
                    f"row as many values as the {whose}'s {role} has."
                )
            raise ValueError(
                f"{who}: the {role} column `{name}` holds {_values(k)} at row {at} (counting from "
                f"0) and {want} at row {set_at}, and every row needs the same number. Give every "
                "row one value, or one for each class."
            )
        for v in values:
            if v is None:
                raise null
            if not math.isfinite(v):
                shown = "NaN" if math.isnan(v) else ("inf" if v > 0 else "-inf")
                raise ValueError(
                    f"{who}: the {role} column `{name}` holds {shown} at row {at} (counting from "
                    "0), and a margin must be finite. Replace it before the build, with 0 where "
                    "the row has no prediction to start from."
                )
        raise AssertionError(  # pragma: no cover
            f"{who}: internal: a margin was refused with nothing wrong, which is a defect in "
            "polars-online. Open an issue at https://github.com/hgilde/polars-online/issues "
            "with this message."
        )


def _values(k: int) -> str:
    """``k`` values, in words: "1 value", "3 values"."""
    return "1 value" if k == 1 else f"{k} values"


def _margin_expr(who: str, role: str, name: str, dt: pl.DataType) -> pl.Expr:
    """The margin as both engines read it: a number as ``Float64``, a
    ``List`` or ``Array`` of numbers as one of ``Float64``."""

    def numeric(d: Any) -> bool:
        return d.is_numeric() and not d.is_decimal()

    if numeric(dt):
        return pl.col(name).cast(pl.Float64)
    if isinstance(dt, pl.List) and numeric(dt.inner):
        return pl.col(name).cast(pl.List(pl.Float64))
    if isinstance(dt, pl.Array) and len(dt.shape) == 1 and numeric(dt.inner):
        return pl.col(name).cast(pl.Array(pl.Float64, dt.size))
    raise WrongTypeError(
        f"{who}: the {role} column {name!r} is {dt}, but a margin is a float or integer "
        "column (one value for each row), or a List or Array of one value for each class. "
        "Cast it to Float64, or to a List of Float64."
    )


def _margin_width_of(obj: Any, read: Callable[[], Any], rows: Callable[[], int]) -> int | None:
    """The values a row of a training set's margin: as recorded on one
    built here, otherwise read from the library (``None`` for no margin)."""
    if hasattr(obj, "_po_margin_width"):
        return obj._po_margin_width
    m = read()
    size = 0 if m is None else int(getattr(m, "size", len(m)))
    n = rows()
    return size // n if size and n else None


def _check_margin_ref(who: str, what: str, cols: _Columns, theirs: int | None) -> None:
    """Refuse a margin on one side alone of a training set and the set
    binned on it: trees fitted around a margin add it to every prediction,
    so a set read without it is off by the margin, and neither library
    warns."""
    role = cols.margin_role
    if theirs is not None and cols.margin is None:
        raise ValueError(
            f"{who}: {what} was built with a {role} of {_values(theirs)} a row, and this "
            f"call gives none. Pass {role}= with these rows' margin, since "
            "trees fitted around a margin add it to every prediction."
        )
    if theirs is None and cols.margin is not None:
        raise ValueError(
            f"{who}: {role}= is given, and {what} was built without one. Build {what} with a "
            f"{role} too, or leave {role}= out here."
        )
    if theirs is not None:
        cols.margin_width = (theirs, None)


def _enum_of(categories: list[str] | None) -> str:
    return "not a pl.Enum" if categories is None else f"a pl.Enum of {categories}"


def _check_ref(
    who: str, what: str, ref: Any, names: Sequence[str] | None, cols: _Columns, binned: str
) -> None:
    """Refuse a ``ref``/``reference`` built on other features, or (one built
    here, which records them) on a ``pl.Enum`` of other categories: its
    codes would be read as the training levels in order."""
    if names is not None and list(names) != cols.names:
        raise ValueError(
            f"{who}: {what} has the features {list(names)}, and this source {cols.names}, but "
            f"{binned} needs the same features in order. Pass the features {what} was built "
            "with, in its order."
        )
    theirs = getattr(ref, "_po_categories", None)
    if theirs is None or len(theirs) != len(cols.names):
        return
    for name, a, b in zip(cols.names, theirs, cols.categories, strict=True):
        if a != b and a is None:
            raise ValueError(
                f"{who}: {what} was built with {name!r} as {_enum_of(a)}, and this source has "
                f"it as {_enum_of(b)}. Cast {name!r} back to the type {what} was built with."
            )
        if a != b:
            raise ValueError(
                f"{who}: {what} was built with {name!r} as {_enum_of(a)}, and this source has "
                f"it as {_enum_of(b)}. A pl.Enum's codes mean its categories in order, so "
                f"{binned} needs the same categories. Cast {name!r} to the pl.Enum {what} was "
                "built with."
            )


def _record(obj: Any, cols: _Columns) -> None:
    """Record each feature's Enum categories, and the margin's values a row,
    on the returned object, for the checks of a later ``ref``/``reference``
    of it."""
    obj._po_categories = list(cols.categories)
    obj._po_margin_width = None if cols.margin_width is None else cols.margin_width[0]


def _whole(v: Any) -> int | None:
    """``v`` as an int where it is a whole number -- any ``numbers.Integral``,
    numpy's integers included, read through ``operator.index`` -- or ``None``
    for a bool or anything else (task 352)."""
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        return None
    return operator.index(v)


def _check_common(
    who: str, max_bin: int, threads: int | None, engine: str
) -> tuple[int, int | None]:
    """``max_bin`` and ``threads`` as Python ints, once each is checked."""
    bins = _whole(max_bin)
    if bins is None:
        raise WrongTypeError(
            f"{who}: max_bin must be a whole number of at least 2, but it is {max_bin!r}. "
            "Pass a whole number of at least 2."
        )
    if bins < 2:
        raise ValueError(
            f"{who}: max_bin must be at least 2, but it is {bins}. Set max_bin to 2 or more."
        )
    library = "XGBoost's" if "xgboost" in who else "LightGBM's"
    count = None if threads is None else _whole(threads)
    if threads is not None and count is None:
        raise WrongTypeError(
            f"{who}: threads must be a whole number of at least 1, or None, but it is "
            f"{threads!r}. Pass a whole number of at least 1, or None for {library} default."
        )
    if count is not None and count < 1:
        raise ValueError(
            f"{who}: threads must be at least 1, or None, but it is {count}. Set threads to "
            f"1 or more, or to None for {library} default."
        )
    if count is not None and count > _native.max_threads():
        raise ValueError(
            f"{who}: threads must be at most {_native.max_threads()}, the most one pool may "
            f"have, but it is {count}. Set threads to at most the machine's cores, since more "
            f"threads than cores run no faster, or to None for {library} default."
        )
    if engine not in _ENGINES:
        raise ValueError(
            f'{who}: engine must be "auto", "native" or "python", but it is {engine!r}. Set '
            "engine to one of those words."
        )
    return bins, count


def _library_path(lib: Any, who: str = "po.gbdt", what: str = "the library") -> str:
    """The file a library's Python package loaded it from."""
    path = getattr(lib, "path", None) or getattr(lib, "_name", None)
    if not path:
        raise ValueError(
            f"{who}: {what}'s Python package does not say which file it loaded, so the native "
            "engine cannot load it. Pass engine='python'."
        )
    return str(path)


@functools.cache
def _xgb_native(path: str) -> Any:
    return _native.GbdtXgboost(path)


@functools.cache
def _lgb_native(path: str, version: str) -> Any:
    return _native.GbdtLightgbm(path, version)


def _holds_handle(lgb: Any) -> bool:
    """Whether LightGBM's ``Dataset`` keeps its native handle in ``_handle``,
    where the native engine hands it over."""
    return hasattr(lgb.Dataset(None), "_handle")


def _native_or_none(
    load: Callable[[], Any], engine: str, what: str, stacklevel: int, who: str
) -> Any:
    if engine == "python":
        return None
    try:
        return load()
    except (ValueError, OSError) as e:
        if engine == "native":
            raise
        # The cause alone: the refusal's first sentence, without its where
        # or its remedy, which this warning gives for itself.
        cause = str(e).split(". ")[0].rstrip(".")
        if cause.startswith(f"{who}: "):
            cause = cause[len(who) + 2 :]
        warnings.warn(
            f"{who}: the native engine cannot load {what} ({cause}), so the build runs on the "
            "Python engine, which grows the same trees more slowly. Pass engine='python' to "
            "choose it and silence this warning.",
            UserWarning,
            stacklevel=stacklevel,
        )
        return None


# ----------------------------------------------------------------- XGBoost


@unstable("polars_online.gbdt.xgboost_matrix")
def xgboost_matrix(
    source: Source,
    *,
    label: str,
    weight: str | None = None,
    features: Sequence[str] | None = None,
    max_bin: int = 256,
    sample_rows: int | None = 200_000,
    ref: xgboost.DMatrix | None = None,
    base_margin: str | None = None,
    threads: int | None = None,
    seed: int = 0,
    engine: Engine = "auto",
) -> xgboost.QuantileDMatrix:
    """Build XGBoost's ``QuantileDMatrix`` from files or a query, binned as the
    rows stream.

    Use it to train XGBoost on Arrow IPC files, Parquet files or a query
    whose result does not fit in memory: ``xgb.train`` takes the matrix as it is, with its
    evaluation sets, early stopping and callbacks. For data already in
    memory, XGBoost's own ``xgb.QuantileDMatrix`` over a numpy array grows
    the same trees as ``sample_rows=None`` here. For LightGBM, use
    :func:`lightgbm_dataset`.

    .. rubric:: Parameters

    ``source``
        The rows: Arrow IPC or Parquet paths (globs taken, all of one
        kind), a ``LazyFrame``, or a ``DataFrame``. A query is run once for
        each pass, its result taken one chunk at a time. The native engine
        reads Arrow IPC files itself, one block of the file at a time.
    ``label``, ``weight``
        The columns of the label, XGBoost's word for the target, and of the
        row weights: numeric or Boolean, with no nulls and finite values,
        and a weight never negative. ``weight`` is ``None`` by default, for
        rows of weight 1.
    ``features``
        The feature columns, in order: numeric, Boolean or ``pl.Enum``
        (the module's *What the columns may hold* says what each may hold).
        ``None``, the default, takes
        every column but the label, the weight and the margin.
    ``max_bin``
        Bins per feature, XGBoost's ``max_bin``, 256 by default as in
        XGBoost. Pass the same to ``xgb.train``.
    ``sample_rows``
        The rows of the sample the cuts are placed from, 200,000 by
        default. ``None`` lets XGBoost sketch every row, and grows XGBoost's
        own trees at about twice the time before the first round (Notes).
        Not read with ``ref``.
    ``ref``
        The training matrix, for a validation or test matrix binned on its
        cuts (``xgb.train`` refuses a ``QuantileDMatrix`` evaluation set
        without it). It must have the same features in order, and a ``ref``
        built here must have each ``pl.Enum`` feature of the same
        categories in order, since a code is read as the training level at
        that position. A ``ref`` built elsewhere is checked by its feature
        names alone. The matrix returned keeps a weak reference to it.
    ``base_margin``
        The column of each row's margin, XGBoost's ``base_margin``: the raw
        score the row's prediction starts from, in place of XGBoost's
        constant ``base_score``, so that the trees fit what the margin
        leaves of the label. ``None`` by default, for no margin. A number a
        row, or for a K-class objective a ``List`` or ``Array`` of K numbers
        a row, one for each class; any numeric type, read as float64 (XGBoost
        keeps it as float32). Each chunk's margin goes to XGBoost with its
        labels, so the margin is never held whole. The cuts do not read it.
        A tree cannot predict outside the range of labels it split on, and
        a margin is added after it. So an online model's prediction as the
        margin carries a level the training rows never reached, and since
        that prediction is out-of-sample, it leaks nothing from the row's
        own label. **Predict with a margin a model trained with one**, or
        every prediction lacks the level the margin carried, and XGBoost
        does not warn. Build the matrix to predict on with its rows' margin
        too, as ``xgb.DMatrix(X, base_margin=m)`` or a matrix built here
        with ``ref`` and ``base_margin``. So a ``ref`` built with a margin
        needs one here, with as many values a row, and a ``ref`` built
        without one needs none.
    ``threads``
        The threads for XGBoost's construction and the conversions.
        ``None``, the default, is XGBoost's default, every core. At most
        1024, or the machine's cores where it has more: more threads than
        cores run no faster.
    ``seed``
        The seed of the sample, 0 by default: the same seed and rows give
        the same cuts.
    ``engine``
        ``"native"``, ``"python"``, or ``"auto"``, the default: the native
        engine, falling back to the Python engine with a warning when the
        native engine cannot load XGBoost. Both grow the same trees. The
        module's *The native engine* says how each reads the rows.

    .. rubric:: Returns

    An ``xgboost.QuantileDMatrix``: the features binned, the labels, the
    weights and the margin. It records each ``pl.Enum`` feature's
    categories and the margin's width, for the checks of a matrix built
    later with it as ``ref``.

    .. rubric:: Raises

    :class:`~polars_online.WrongTypeError`
        ``max_bin``, ``threads``, ``sample_rows`` or ``seed`` is not a
        whole number; or a feature, the label, the weight or the margin is of
        a type XGBoost does not read, a ``pl.Categorical`` among them. It is both
        a ``TypeError`` and a ``ValueError``.
    :class:`~polars_online.ColumnNotFoundError`
        A column named is not in the source. It is both Polars'
        ``ColumnNotFoundError`` and a ``ValueError``.
    ``TypeError``
        ``ref`` is not an ``xgboost.DMatrix``, ``source`` is of another
        type, or ``features`` is one name in place of a list.
    ``FileNotFoundError``
        A glob of local files matches nothing.
    ``ValueError``
        When, each named in the message:

        - ``max_bin`` below 2, ``threads`` or ``sample_rows`` below 1, a
          ``threads`` above its bound, a ``seed`` outside 0 to ``2**64 - 1``,
          or an ``engine`` not one of the three;
        - a source that names no files, mixes Arrow IPC and Parquet, or names a
          file whose extension says neither;
        - a column named twice, or as both a feature and the label, weight or
          margin, or an empty ``features``;
        - with ``features`` left out, a source with no column besides the
          label, the weight and the margin;
        - a null or non-finite label or weight, or a negative weight;
        - a null or non-finite margin, or a row whose number of margin values
          differs from the first row's or from the ``ref``'s margin;
        - an empty source, or one whose passes give different rows;
        - a ``ref`` binned on other features, or built here on a ``pl.Enum``
          of other categories;
        - a ``ref`` built with a margin and no ``base_margin`` here, or
          ``base_margin`` here and a ``ref`` built without one;
        - under ``engine="native"``, an XGBoost the native engine cannot load.

    ``UserWarning``
        Under ``engine="auto"``, the native engine cannot load XGBoost, and
        the Python engine builds the matrix.
    :class:`~polars_online.UnstableWarning`
        Under ``POLARS_ONLINE_WARN_UNSTABLE=1``, on each call (the module's
        warning says why).

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_, and writes it
    to Arrow IPC files, which the native engine reads itself:

    .. code-block:: python

        import xgboost as xgb

        df.head(300).write_ipc("train.arrow")              # the example data, as two Arrow files
        df.tail(100).write_ipc("valid.arrow")
        features = ["x0", "x1", "x2", "signal_a", "signal_b"]
        train = po.gbdt.xgboost_matrix("train.arrow", label="y", features=features)
        valid = po.gbdt.xgboost_matrix("valid.arrow", label="y", features=features, ref=train)
        booster = xgb.train({"tree_method": "hist", "max_bin": 256}, train, 500,
                            evals=[(valid, "valid")], early_stopping_rounds=20, verbose_eval=False)
        rows, cols = train.num_row(), train.num_col()
        print(f"{rows} rows of {cols} features, {valid.num_row()} to validate")

    .. code-block:: text

        300 rows of 5 features, 100 to validate

    .. rubric:: See also

    - :func:`lightgbm_dataset`, LightGBM's training input from the same
      sources.
    - `Training XGBoost and LightGBM
      <https://github.com/hgilde/polars-online/blob/main/docs/GBDT.md>`_,
      the guide: training around an online model's prediction as the
      margin, and serving with it.

    .. rubric:: Notes

    XGBoost places each feature's bins at its quantiles and reads the input
    in passes: it counts the values, sketches the quantiles over every row,
    indexes each value into its bin and builds the column matrix.

    **The cuts.** With ``sample_rows``, one pass first draws a sample of
    that many rows. Row ``i`` has a hash ``h``, SplitMix64 of ``i`` and
    ``seed``, and the priority ``w / u``, where ``w`` is its weight (1
    without ``weight``) and ``u = (2h + 1) / 2**65``. The ``sample_rows``
    rows of largest priority are kept. Without weights these are the rows
    of smallest hash, a uniform sample. With weights it is priority
    sampling: a row heavier than ``tau``, the largest priority not kept, is
    always kept and counts its own weight, and any other kept row counts
    ``tau``. Summed over any set of rows, these counts estimate the set's
    weight without bias, so the sample's weighted quantiles estimate the
    source's. A row of weight 0 counts for nothing and places no cut. The
    sample is drawn in proportion to the weights, not uniformly, because a
    uniform sample loses heavy rows. Under a weight of 1e9 on one row in a
    million, a uniform sample of 200,000 rows misses that row four times in
    five, and its cuts ignore it.

    Each feature's bin edges sit at the sample's weighted quantiles. A
    feature with at most ``max_bin`` distinct sample values gets a bin per
    value. Any other gets ``max_bin`` bins of equal weight. Edge ``j`` is
    the first sorted sample value whose running weight exceeds
    ``j / max_bin`` of the total. Without weights that is the value of rank
    ``floor(j m / max_bin)`` for ``m`` values. At the quantile ``q``, a
    cut's share of the weight then errs by about
    ``sqrt(q (1 - q) / sample_rows)``, 0.11% at the median for 200,000 rows.
    Over ten seeds on a million rows, its standard deviation measured 0.29%
    to 0.46% with 20,000 rows, where the formula gives 0.35%, and 0.04% with
    200,000 rows under weight 50 on a sixth of them. The cuts reach XGBoost
    as a ``ref``, which removes its sketch pass. The sample is the same rows
    however the source is batched and whichever engine reads it.
    ``sample_rows=None`` lets XGBoost sketch every row, and grows XGBoost's
    own trees.

    **Measured** as :mod:`polars_online.gbdt` describes, on 10 million rows
    of 50 features: the time before the first round, and the peak memory.
    Reading ahead, Polars holds 7.2 GB at the peak on the query route, and
    `Bound a run's memory
    <https://github.com/hgilde/polars-online/blob/main/docs/how-to/bound-memory.md#cut-what-polars-reads-ahead>`_
    cuts it.

    .. list-table::
       :header-rows: 1
       :widths: 60 20 20

       * - route
         - seconds
         - peak, GB
       * - this, from Arrow IPC files
         - 4.0
         - 3.6
       * - this, from a ``LazyFrame`` over them
         - 4.2
         - 7.2
       * - this, ``engine="python"``
         - 5.5
         - 7.1
       * - this, ``sample_rows=None``
         - 8.0
         - 3.6
       * - XGBoost's own ``DataIter`` over Polars' chunks
         - 8.7
         - 4.8
       * - ``collect()`` and numpy
         - 7.0
         - 8.7
    """
    import xgboost as xgb

    who = "po.gbdt.xgboost_matrix"
    max_bin, threads = _check_common(who, max_bin, threads, engine)
    if sample_rows is not None and _whole(sample_rows) is None:
        raise WrongTypeError(
            f"{who}: sample_rows must be a whole number of at least 1, or None, but it is "
            f"{sample_rows!r}. Pass a whole number of at least 1, or None for XGBoost's own "
            "sketch of every row."
        )
    sample_rows = None if sample_rows is None else _whole(sample_rows)
    if sample_rows is not None and sample_rows < 1:
        raise ValueError(
            f"{who}: sample_rows must be at least 1, or None, but it is {sample_rows}. Set "
            "sample_rows to 1 or more, or to None for XGBoost's own sketch of every row."
        )
    if _whole(seed) is None:
        raise WrongTypeError(
            f"{who}: seed must be a whole number from 0 to 2**64 - 1, but it is {seed!r}. "
            "Pass a whole number, such as 0."
        )
    seed = _whole(seed)  # type: ignore[assignment]
    if not 0 <= seed < 2**64:
        raise ValueError(
            f"{who}: seed must be from 0 to 2**64 - 1, but it is {seed}. Set seed to a whole "
            "number in that range, such as 0."
        )
    if ref is not None and not isinstance(ref, xgb.DMatrix):
        raise TypeError(
            f"{who}: ref must be an xgboost.DMatrix, but its type is {type(ref).__name__}. "
            "Pass the training matrix xgboost_matrix returned."
        )
    plan, files = _plan(source, who)
    cols = _Columns(
        plan,
        label=label,
        weight=weight,
        features=features,
        who=who,
        margin=base_margin,
        margin_role="base_margin",
    )
    if ref is not None:
        _check_ref(who, "ref", ref, ref.feature_names, cols, "a matrix binned on another's cuts")
        theirs = _margin_width_of(ref, ref.get_base_margin, ref.num_row)
        _check_margin_ref(who, "ref", cols, theirs)
    native = _native_or_none(
        lambda: _xgb_native(_library_path(xgb.core._LIB, who, "XGBoost")),
        engine,
        "XGBoost",
        4,
        who,
    )
    if native is not None:
        handle, stats = native.matrix(
            cols.batches,
            cols.names,
            cols.kinds,
            cols.levels,
            cols.label,
            cols.weight,
            max_bin,
            threads or 0,
            None if ref is not None else sample_rows,
            seed,
            None if ref is None else ref.handle.value,
            _READ_AHEAD,
            None if files is None else (files, cols.categories),
            cols.margin,
            None if cols.margin_width is None else cols.margin_width[0],
        )
        _LAST_STATS.clear()
        _LAST_STATS.update(stats)
        if "margin_width" in stats:
            cols.margin_width = (int(stats["margin_width"]), None)
        # The constructor's branch for a handle made elsewhere: the matrix
        # owns it from here and frees it once, when it is collected.
        out = xgb.QuantileDMatrix(ctypes.c_void_p(handle), max_bin=max_bin, nthread=threads)
    else:
        out = _xgb_python(xgb, cols, max_bin, sample_rows, ref, threads, seed)
    if ref is not None:
        out.ref = weakref.ref(ref)
    _record(out, cols)
    return out


def _xgb_python(
    xgb: Any,
    cols: _Columns,
    max_bin: int,
    sample_rows: int | None,
    ref: Any,
    threads: int | None,
    seed: int,
) -> Any:
    """The Python engine: numpy batches into a ``DataIter``, the cuts from
    the same sample as the native engine's."""
    import numpy as np

    refm = ref
    if ref is None and sample_rows is not None:
        sampler = _native.GbdtSampler(
            cols.names,
            cols.kinds,
            cols.levels,
            cols.label,
            cols.weight,
            sample_rows,
            seed,
            threads or 0,
            cols.margin,
        )
        for frame in cols.batches():
            sampler.push(frame)
        if sampler.rows_seen == 0:
            raise ValueError(
                "po.gbdt.xgboost_matrix: the source has no rows. Pass a source with rows."
            )
        data, rows, p = sampler.ref_matrix(max_bin)
        planted = np.frombuffer(data, dtype="<f4").reshape(rows, p)
        refm = xgb.DMatrix(
            planted,
            feature_names=cols.names,
            feature_types=cols.xgb_types,
            enable_categorical=True,
            nthread=threads,
        )

    class Batches(xgb.DataIter):
        def __init__(self) -> None:
            self._it: Iterator[pl.DataFrame] | None = None
            self._row = 0
            self._rows: int | None = None
            super().__init__(release_data=True)

        def reset(self) -> None:
            self._it = None
            self._row = 0

        def next(self, input_data: Callable[..., None]) -> bool:
            if self._it is None:
                self._it = cols.batches()
            frame = next(self._it, None)
            if frame is None:
                if self._row == 0:
                    raise ValueError(
                        "po.gbdt.xgboost_matrix: the source has no rows. Pass a source with rows."
                    )
                if self._rows is not None and self._rows != self._row:
                    raise ValueError(
                        "po.gbdt.xgboost_matrix: the source gave "
                        f"{self._rows} rows in one pass and "
                        f"{self._row} in another, but XGBoost reads it more than once, and "
                        "every pass must give the same rows. Pass a source that reads the "
                        "same rows each time, such as files or a frame."
                    )
                self._rows = self._row
                return False
            if frame.height == 0:
                return self.next(input_data)
            cols.check_meta(frame, self._row)
            meta: dict[str, Any] = {"label": frame[cols.label].to_numpy()}
            if cols.weight is not None:
                meta["weight"] = frame[cols.weight].to_numpy()
            if cols.margin is not None:
                meta["base_margin"] = cols.margin_numpy(frame, self._row)
            input_data(
                data=frame.select(cols.names).to_numpy(),
                feature_names=cols.names,
                feature_types=cols.xgb_types,
                **meta,
            )
            self._row += frame.height
            return True

    return xgb.QuantileDMatrix(
        Batches(), ref=refm, max_bin=max_bin, nthread=threads, enable_categorical=True
    )


# ---------------------------------------------------------------- LightGBM

#: Dataset parameters this module sets itself, from its own arguments.
_LGB_OWN = {
    "max_bin": "max_bin",
    "categorical_feature": "categorical",
    "cat_feature": "categorical",
    "categorical_column": "categorical",
    "cat_column": "categorical",
    "categorical_features": "categorical",
    "num_threads": "threads",
    "num_thread": "threads",
    "nthread": "threads",
    "nthreads": "threads",
    "n_jobs": "threads",
}


def _lgb_param_str(params: dict[str, Any]) -> str:
    """LightGBM's ``key=value`` parameter string, as its Python package
    writes one."""
    pairs = []
    for k, v in params.items():
        if isinstance(v, (list, tuple, set)):
            pairs.append(f"{k}={','.join(map(str, v))}")
        elif v is not None:
            pairs.append(f"{k}={v}")
    return " ".join(pairs)


def _available_memory() -> int:
    return int(_native.gbdt_available_memory())


@unstable("polars_online.gbdt.lightgbm_dataset")
def lightgbm_dataset(
    source: Source,
    *,
    label: str,
    weight: str | None = None,
    features: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    max_bin: int = 255,
    params: dict[str, Any] | None = None,
    reference: lightgbm.Dataset | None = None,
    init_score: str | None = None,
    in_memory: bool | Literal["auto"] = "auto",
    threads: int | None = None,
    engine: Engine = "auto",
) -> lightgbm.Dataset:
    """Build a constructed LightGBM ``Dataset`` from files or a query, binned
    on LightGBM's own sample.

    Use it to train LightGBM on Arrow IPC files, Parquet files or a query
    whose result does not fit in memory: ``lgb.train`` takes the dataset as it is, with its
    validation sets, early stopping and callbacks. It grows the trees
    LightGBM's own route grows from a float32 array of the data. For
    XGBoost, use :func:`xgboost_matrix`.

    .. rubric:: Parameters

    ``source``
        The rows: Arrow IPC or Parquet paths (globs taken, all of one
        kind), a ``LazyFrame``, or a ``DataFrame``. A query is run once for
        each pass, its result taken one chunk at a time. The native engine
        reads Arrow IPC files itself, one block of the file at a time.
    ``label``, ``weight``
        The columns of the label, LightGBM's word for the target, and of
        the row weights: numeric or Boolean, with no nulls and finite
        values, and a weight never negative. ``weight`` is ``None`` by
        default, for rows of weight 1.
    ``features``
        The feature columns, in order: numeric, Boolean or ``pl.Enum``
        (the module's *What the columns may hold* says what each may hold).
        ``None``, the default, takes
        every column but the label, the weight and the margin.
    ``categorical``
        The categorical features, ``pl.Enum``, integer or Boolean columns (a
        code below 0 is missing, as LightGBM reads it). ``None``, the
        default, takes every ``pl.Enum`` feature. A ``pl.Enum`` left out
        goes in as its codes, a number.
    ``max_bin``
        Bins per feature, LightGBM's ``max_bin``, 255 by default as in
        LightGBM.
    ``params``
        Other dataset parameters, as ``lightgbm.Dataset(params=)`` takes
        them: ``verbose``, ``min_data_in_bin``, ``bin_construct_sample_cnt``,
        ``data_random_seed``, ``use_missing``, ``zero_as_missing``, ... Not
        ``max_bin``, ``categorical_feature`` or ``num_threads``, which are
        this function's own arguments. ``None`` by default. Training may
        not change a dataset parameter, since the dataset is constructed and
        its raw data is not kept: LightGBM refuses, as it does for any
        constructed dataset.
    ``reference``
        The training dataset, for a validation dataset binned on its bins,
        built in one pass. It must have the same features in order, when it
        records its feature names. A ``reference`` built here must have each
        ``pl.Enum`` feature of the same categories in order, since a code
        is read as the training level at that position. A ``reference``
        built elsewhere is checked by its feature names alone. Given a
        ``reference``, this call's ``max_bin``, ``params`` and
        ``categorical`` are checked as usual but not used: the dataset takes
        the reference's bins, parameters and categorical features.
    ``init_score``
        The column of each row's margin, LightGBM's ``init_score``: the raw
        score the row's prediction starts from, so that the trees fit what
        the margin leaves of the label. LightGBM then skips
        ``boost_from_average``, its own constant start. ``None`` by default,
        for no margin. A number a row, or for a K-class objective a ``List``
        or ``Array`` of K numbers a row, one for each class; any numeric
        type, read as float64. The native engine's two passes push each
        chunk's margin with its rows, so the margin is never held whole.
        ``in_memory`` holds every row's, ``n · K`` float64 values for ``n``
        rows, as it holds the labels. ``engine="python"`` holds them too,
        because LightGBM's Python ``Dataset`` takes ``init_score`` whole:
        80 MB for 10 million rows of one class, and K times that for K
        classes. The bins do not read it. A tree cannot predict outside the
        range of labels it split on, and a margin is added after it. So an
        online model's prediction as the margin carries a level the
        training rows never reached, and since that prediction is
        out-of-sample, it leaks nothing from the row's own label.
        **Predict with a margin a model trained with one**, or every
        prediction lacks the level the margin carried, and LightGBM does
        not warn: add the rows' margin to
        ``booster.predict(X, raw_score=True)``. So a ``reference`` built
        with a margin needs one here, with as many values a row, and a
        ``reference`` built without one needs none.
    ``in_memory``
        Whether to hand LightGBM the whole stream at once, held in memory:
        as Arrow under the native engine, as a float32 numpy array under the
        Python one. ``True`` always does, ``False`` never does, and
        ``"auto"``, the default, does when the features as float32 take at
        most a quarter of the memory available and, under the native
        engine, the LightGBM loaded takes an Arrow stream (4.7.0 on).
    ``threads``
        The threads for LightGBM's construction and the conversions.
        ``None``, the default, is LightGBM's default, every core. At most
        1024, or the machine's cores where it has more: more threads than
        cores run no faster.
    ``engine``
        ``"native"``, ``"python"``, or ``"auto"``, the default: the native
        engine, falling back to the Python engine with a warning when the
        native engine cannot load LightGBM. Both grow the same trees.

    .. rubric:: Returns

    A constructed ``lightgbm.Dataset``: the features binned, the labels,
    the weights and the margin (``init_score``), with its raw data freed. It records
    each ``pl.Enum`` feature's categories and the margin's width, for the
    checks of a dataset built later with it as ``reference``.

    .. rubric:: Raises

    :class:`~polars_online.WrongTypeError`
        ``max_bin`` or ``threads`` is not a whole number; a feature, the
        label, the weight or the margin is of a type LightGBM does not
        read, a ``pl.Categorical`` among them; or ``categorical`` names a
        float column. It is both a ``TypeError`` and a ``ValueError``.
    :class:`~polars_online.ColumnNotFoundError`
        A column named is not in the source. It is both Polars'
        ``ColumnNotFoundError`` and a ``ValueError``.
    ``TypeError``
        ``reference`` is not a ``lightgbm.Dataset`` (one not yet
        constructed is constructed here), ``source`` is of another type, or
        ``features`` or ``categorical`` is one name in place of a list.
    ``FileNotFoundError``
        A glob of local files matches nothing.
    ``ValueError``
        When, each named in the message:

        - ``max_bin`` below 2, ``threads`` below 1 or above its bound, an ``engine`` not one of
          the three, or an ``in_memory`` that is not ``True``, ``False`` or
          ``"auto"``;
        - a parameter in ``params`` that is this function's own argument;
        - a source that names no files, mixes Arrow IPC and Parquet, or
          names a file whose extension says neither;
        - a column named twice, or as both a feature and the label, weight
          or margin, or an empty ``features``;
        - with ``features`` left out, a source with no column besides the
          label, the weight and the margin;
        - a ``categorical`` name that is not a feature;
        - a null or non-finite label or weight, or a negative weight;
        - a null or non-finite margin, or a row whose number of margin
          values differs from the first row's or from the ``reference``'s
          margin;
        - an empty source, more rows than LightGBM's 32-bit row count
          holds, or a source whose passes give different rows;
        - a ``reference`` that records other features, or built here on a
          ``pl.Enum`` of other categories;
        - a ``reference`` built with a margin and no ``init_score`` here,
          or ``init_score`` here and a ``reference`` built without one;
        - under ``engine="native"``, a LightGBM the native engine cannot
          load;
        - ``in_memory=True`` under the native engine with a LightGBM before
          4.7.0, which takes no Arrow stream (``engine="auto"`` does not
          fall back on it).
    ``RuntimeError``
        Under the Python engine, LightGBM asks for the rows out of order,
        which a source read in order cannot give: pass ``in_memory=True``.
    ``UserWarning``
        Under ``engine="auto"``, the native engine cannot load LightGBM,
        and the Python engine builds the dataset.
    :class:`~polars_online.UnstableWarning`
        Under ``POLARS_ONLINE_WARN_UNSTABLE=1``, on each call (the module's
        warning says why).

    .. rubric:: Examples

    This code reads ``ticks/*.parquet`` and ``today.parquet`` from
    `Example data
    <https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_, the training
    rows by a glob:

    .. code-block:: python

        import lightgbm as lgb

        features = ["x0", "x1", "x2", "signal_a", "signal_b"]
        train = po.gbdt.lightgbm_dataset("ticks/*.parquet", label="y", features=features,
                                         params={"verbose": -1})
        valid = po.gbdt.lightgbm_dataset("today.parquet", label="y", features=features,
                                         reference=train)
        booster = lgb.train({"objective": "regression", "verbose": -1}, train, 500,
                            valid_sets=[valid],
                            callbacks=[lgb.early_stopping(20, verbose=False)])
        rows, cols = train.num_data(), train.num_feature()
        print(f"{rows} rows of {cols} features, {valid.num_data()} to validate")

    .. code-block:: text

        400 rows of 5 features, 400 to validate

    .. rubric:: See also

    - :func:`xgboost_matrix`, XGBoost's training input from the same
      sources.
    - `Training XGBoost and LightGBM
      <https://github.com/hgilde/polars-online/blob/main/docs/GBDT.md>`_,
      the guide: training around an online model's prediction as the
      margin, and serving with it.

    .. rubric:: Notes

    LightGBM places each feature's bins from a sample of
    ``bin_construct_sample_cnt`` rows (200,000 by default), drawn from the
    row count and ``data_random_seed``, and then pushes every value into its
    bin. Here the source is counted (``select(pl.len())``), a first pass
    gathers the rows LightGBM samples, and a second pushes every row as
    float32, built in parallel: two passes, and the data never held. With
    ``in_memory``, one pass holds the features as float32 Arrow and hands
    LightGBM the whole stream, which it pushes column by column. Both routes
    hand LightGBM the same float32 values and grow the same trees, which are
    LightGBM's own from a float32 array of the data.

    **Weights do not move the bins**, in LightGBM or here. LightGBM's
    sample is uniform, and neither ``LGBM_DatasetCreateFromSampledColumn``
    nor the ``BinMapper::FindBin`` it calls takes a weight, so its bins are
    those of the unweighted rows. :func:`xgboost_matrix` weights its cuts,
    because XGBoost's own sketch does: each builds what its library builds.
    With weight 50 on the values above 1.0 of 50,000 normal ones and
    ``max_bin=16``, LightGBM's trees split at the same 14 bounds with the
    weights as without, 2 of them above 1.0, where 14 of XGBoost's 15 cuts
    are.

    **Measured** as :mod:`polars_online.gbdt` describes, on 10 million rows
    of 50 features from Arrow IPC files: the time before the first round,
    and the peak memory.

    .. list-table::
       :header-rows: 1
       :widths: 60 20 20

       * - route
         - seconds
         - peak, GB
       * - this, streamed in two passes
         - 1.68
         - 1.7
       * - this, ``in_memory``
         - 1.45
         - 2.9
       * - LightGBM's own route from a Polars frame in memory
         - 1.54
         - 8.9
       * - LightGBM's ``Sequence``
         - 2.17
         - 2.7
    """
    import lightgbm as lgb

    who = "po.gbdt.lightgbm_dataset"
    max_bin, threads = _check_common(who, max_bin, threads, engine)
    # By type: `1 in (True, False)` holds, and 1 is not an answer.
    if not isinstance(in_memory, bool) and in_memory != "auto":
        raise ValueError(
            f"{who}: in_memory must be True, False or 'auto', but it is {in_memory!r}. Set "
            "in_memory to one of those."
        )
    if reference is not None:
        if not isinstance(reference, lgb.Dataset):
            raise TypeError(
                f"{who}: reference must be a lightgbm.Dataset, but its type is "
                f"{type(reference).__name__}. Pass the training set lightgbm_dataset returned."
            )
        reference.construct()
    given = dict(params or {})
    for key, arg in _LGB_OWN.items():
        if key in given:
            raise ValueError(
                f"{who}: params names {key!r}, which this call takes as an argument of its "
                f"own. Pass it as {arg}= instead."
            )
    plan, files = _plan(source, who)
    cols = _Columns(
        plan,
        label=label,
        weight=weight,
        features=features,
        who=who,
        margin=init_score,
        margin_role="init_score",
    )
    if reference is not None:
        # LightGBM's own dataset records its names when given them, and one
        # built here always does.
        named = reference.feature_name
        _check_ref(
            who,
            "reference",
            reference,
            named if isinstance(named, (list, tuple)) else None,
            cols,
            "a dataset binned on another's bins",
        )
        theirs = _margin_width_of(reference, reference.get_init_score, reference.num_data)
        _check_margin_ref(who, "reference", cols, theirs)
    if categorical is None:
        cats = [j for j, k in enumerate(cols.kinds) if k == "category"]
    else:
        if isinstance(categorical, str):
            raise TypeError(
                f"{who}: categorical must be a list of column names, but it is the one name "
                f"{categorical!r}. Pass it in a list, as [{categorical!r}]."
            )
        cats = []
        for c in categorical:
            if c not in cols.names:
                raise ValueError(
                    f"{who}: categorical names {c!r}, but it is not a feature. Name one of "
                    "the features."
                )
            j = cols.names.index(c)
            if cols.kinds[j] not in ("category", "int", "bool"):
                raise WrongTypeError(
                    f"{who}: categorical names {c!r}, a float column, but LightGBM reads a "
                    "categorical feature as integer codes. Cast it to an integer or a pl.Enum, "
                    "or leave it out of categorical."
                )
            cats.append(j)
        cats = sorted(set(cats))
    dparams: dict[str, Any] = {"max_bin": max_bin, **given}
    if cats:
        # The key LightGBM's own `Dataset` records the indices under.
        dparams["categorical_column"] = cats
    # What `Dataset.categorical_feature` holds, which `lgb.train` compares a
    # validation set's with its training set's.
    cat_attr: Any = cats if cats else "auto"
    if reference is not None:
        # A validation set takes its training set's bins, parameters and
        # categorical features.
        dparams = dict(reference.params or {})
        cat_attr = reference.categorical_feature
    build = dict(dparams)
    if threads is not None:
        build["num_threads"] = threads
    rows = int(cols.plan.select(pl.len()).collect().item())
    if rows == 0:
        raise ValueError(f"{who}: the source has no rows. Pass a source with rows.")

    def load() -> Any:
        # Checked as part of the load, so that engine="auto" falls back to
        # the Python engine on it as on any other (task 352), where it was
        # raised after the load and refused every engine.
        if not _holds_handle(lgb):
            raise ValueError(
                f"{who}: the native engine does not support lightgbm {lgb.__version__}, whose "
                "Dataset holds its native handle where the engine does not look. Pass "
                "engine='python', which builds through LightGBM's Python API."
            )
        return _lgb_native(_library_path(lgb.basic._LIB, who, "LightGBM"), lgb.__version__)

    native = _native_or_none(load, engine, "LightGBM", 4, who)
    if in_memory == "auto":
        fits = rows * len(cols.names) * 4 * 4 <= _available_memory()
        memory = fits and (native is None or native.has_arrow)
    else:
        memory = bool(in_memory)
    if native is not None:
        ref_handle = None
        if reference is not None:
            # Constructed above, so it holds its handle.
            assert reference._handle is not None
            ref_handle = reference._handle.value
        handle, stats = native.dataset(
            cols.batches,
            cols.names,
            cols.kinds,
            cols.levels,
            cols.label,
            cols.weight,
            _lgb_param_str(build),
            threads or 0,
            rows,
            memory,
            ref_handle,
            _READ_AHEAD,
            None if files is None else (files, cols.categories),
            cols.margin,
            None if cols.margin_width is None else cols.margin_width[0],
        )
        _LAST_STATS.clear()
        _LAST_STATS.update(stats)
        if "margin_width" in stats:
            cols.margin_width = (int(stats["margin_width"]), None)
        ds = lgb.Dataset(None, reference=reference, params=dparams, free_raw_data=True)
        # The dataset owns the handle from here and frees it once
        # (`Dataset._free_handle`, on collection).
        ds._handle = ctypes.c_void_p(handle)
        ds.feature_name = list(cols.names)
        ds.categorical_feature = cat_attr
    else:
        ds = _lgb_python(lgb, cols, rows, dparams, build, cat_attr, reference, memory)
    _record(ds, cols)
    return ds


def _lgb_python(
    lgb: Any,
    cols: _Columns,
    rows: int,
    dparams: dict[str, Any],
    build: dict[str, Any],
    cat_attr: Any,
    reference: Any,
    memory: bool,
) -> Any:
    """The Python engine: a float32 numpy matrix in memory, or a
    ``lightgbm.Sequence`` over the batches, LightGBM's sample rows gathered
    in a first pass."""
    import numpy as np

    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    as_f32 = [pl.col(c).cast(pl.Float32) for c in cols.names]

    def meta(frame: pl.DataFrame, row: int) -> None:
        cols.check_meta(frame, row)
        labels.append(frame[cols.label].to_numpy())
        if cols.weight is not None:
            weights.append(frame[cols.weight].to_numpy())
        if cols.margin is not None and frame.height:
            margins.append(cols.margin_numpy(frame, row))

    if memory:
        frames = []
        row = 0
        for frame in cols.batches():
            meta(frame, row)
            frames.append(frame.select(as_f32))
            row += frame.height
        if row != rows:
            raise ValueError(
                "po.gbdt.lightgbm_dataset: the source was counted at "
                f"{rows} rows and a pass over it gave {row}, "
                "but the build reads it more than once, to count its rows and then to read "
                "them, and every pass must give the same rows. Pass a source that reads the "
                "same rows each time, such as files or a frame."
            )
        data: Any = pl.concat(frames).to_numpy(order="c")
    else:
        probe = lgb.Dataset(None, params=build)
        sample = (
            np.sort(probe._create_sample_indices(total_nrow=rows))
            if reference is None
            else np.empty(0, dtype=np.int32)
        )
        gathered: dict[int, np.ndarray] = {}
        row = 0
        for frame in cols.batches():
            meta(frame, row)
            lo, hi = np.searchsorted(sample, [row, row + frame.height])
            if hi > lo:
                block = frame.select(as_f32).to_numpy(order="c")
                for i in sample[lo:hi]:
                    # float64, as LightGBM's sample takes it: the float32
                    # values, exactly.
                    gathered[int(i)] = block[int(i) - row].astype(np.float64)
            row += frame.height
        if row != rows:
            raise ValueError(
                "po.gbdt.lightgbm_dataset: the source was counted at "
                f"{rows} rows and a pass over it gave {row}, "
                "but the build reads it more than once, to count its rows and then to read "
                "them, and every pass must give the same rows. Pass a source that reads the "
                "same rows each time, such as files or a frame."
            )
        data = _lgb_rows(lgb, cols, rows, gathered, as_f32)
    label = np.concatenate(labels).astype(np.float32)
    weight = np.concatenate(weights).astype(np.float32) if cols.weight is not None else None
    # LightGBM's Python API takes the initial scores whole: n * K float64
    # values, held here until the dataset is built.
    scores = np.concatenate(margins) if margins else None
    ds = lgb.Dataset(
        data,
        label=label,
        weight=weight,
        init_score=scores,
        reference=reference,
        feature_name=list(cols.names),
        categorical_feature=cat_attr,
        # The categorical features go by the argument; LightGBM ignores the
        # parameter beside it, with a warning.
        params={k: v for k, v in build.items() if k != "categorical_column"},
        free_raw_data=True,
    ).construct()
    ds.params = dparams
    return ds


def _lgb_rows(
    lgb: Any, cols: _Columns, rows: int, gathered: dict[int, Any], as_f32: list[pl.Expr]
) -> Any:
    """A ``lightgbm.Sequence`` answering LightGBM's single-row requests from
    the rows gathered in the first pass, and its in-order batch requests
    from a second pass over the source."""
    import numpy as np

    class Rows(lgb.Sequence):
        batch_size = _CHUNK

        def __init__(self) -> None:
            self._it: Iterator[pl.DataFrame] | None = None
            self._buf = np.empty((0, len(cols.names)), dtype=np.float32)
            self._at = 0

        def __len__(self) -> int:
            return rows

        def __getitem__(self, idx: Any) -> Any:
            if not isinstance(idx, slice):
                return gathered[int(idx)]
            start, stop, _ = idx.indices(rows)
            if start != self._at:
                raise RuntimeError(
                    "po.gbdt.lightgbm_dataset: LightGBM asked for the rows out of order, and "
                    "the source can be read only in order. Pass in_memory=True, which holds the "
                    "rows in memory."
                )
            if self._it is None:
                self._it = cols.batches()
            while self._buf.shape[0] < stop - start:
                frame = next(self._it, None)
                if frame is None:
                    raise ValueError(
                        "po.gbdt.lightgbm_dataset: the source was counted at "
                        f"{rows} rows and a pass over it gave {self._at + self._buf.shape[0]}, "
                        "but the build reads it more than once, and every pass must give the "
                        "same rows. Pass a source that reads the same rows each time, such as "
                        "files or a frame."
                    )
                block = frame.select(as_f32).to_numpy(order="c")
                self._buf = np.concatenate([self._buf, block]) if self._buf.size else block
            out, self._buf = self._buf[: stop - start], self._buf[stop - start :]
            self._at = stop
            if stop == rows:
                # The last rows LightGBM asks for: a pass that goes on past
                # them gave more rows than were counted, which LightGBM,
                # reading only as many as counted, would never see.
                more = self._buf.shape[0] + sum(f.height for f in self._it)
                if more:
                    raise ValueError(
                        "po.gbdt.lightgbm_dataset: the source was counted at "
                        f"{rows} rows and a pass over it gave {rows + more}, but the build "
                        "reads it more than once, and every pass must give the same rows. Pass "
                        "a source that reads the same rows each time, such as files or a frame."
                    )
            return out

    return Rows()
