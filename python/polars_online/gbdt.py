"""XGBoost's and LightGBM's training inputs, built from Arrow files, Parquet
files or a ``LazyFrame`` as fast as the libraries go.

Both libraries train in two stages. *Binning* places each feature's bins and
puts every value into one; *boosting* grows trees from per-bin sums and never
reads a raw value again. This module does the first stage from data that
streams, in Rust, and hands back the library's own training object, so that
``xgb.train`` and ``lgb.train`` run unchanged, with their evaluation sets,
early stopping and callbacks. Here the training rows are the example data's
two parquet files, read by a glob:

.. code-block:: python

    import xgboost as xgb

    features = ["x0", "x1", "x2", "signal_a", "signal_b"]
    train = po.gbdt.xgboost_matrix("ticks/*.parquet", label="y", features=features)
    valid = po.gbdt.xgboost_matrix("today.parquet", label="y", features=features, ref=train)
    booster = xgb.train({"tree_method": "hist", "max_bin": 256}, train, 500,
                        evals=[(valid, "valid")], early_stopping_rounds=20, verbose_eval=False)

- :func:`xgboost_matrix` returns an ``xgboost.QuantileDMatrix``. XGBoost reads
  its input four times, and the second pass sketches every feature's
  quantiles over every row; this places the cuts from a uniform sample of
  ``sample_rows`` rows instead, which removes that pass, about half the time
  before the first round. The trees then differ from XGBoost's own sketch
  within the spread a sample's seed already gives (``docs/GBDT-ADAPTER.md``
  §4.5); ``sample_rows=None`` keeps XGBoost's sketch and its own trees.
- :func:`lightgbm_dataset` returns a constructed ``lightgbm.Dataset``, binned
  on LightGBM's own sample: a first pass gathers the rows LightGBM samples, a
  second pushes every row as float32; or, when the data fits, the whole
  stream goes in as Arrow.

**The native engine** reads Arrow IPC files itself, a record batch at a
time, and any other source through Polars' streaming engine
(``collect_batches``). It hands each batch's Arrow buffers to the library's C
API with no copy: nulls go by Arrow validity, not as a NaN copy. The library is loaded
from the user's own installed package, by the path that package loaded it
from: nothing is linked into this package, whose only dependency stays
``polars``. ``engine="python"`` builds through the libraries' Python APIs
instead, numpy batches into an ``xgboost.DataIter`` with the same sample's
cuts, and a ``lightgbm.Sequence``; ``engine="auto"`` falls back to it, with a
warning, when the native engine cannot load the library. Both engines grow
the same trees.

**What the columns may hold.** Every feature is numeric (any integer or float
dtype), Boolean, or a ``pl.Enum``, whose codes go in as a categorical
feature: their meaning is fixed by the Enum's categories, so a model is read
the same way at prediction (pass the codes, ``pl.col(c).to_physical()``). A
``pl.Categorical`` is refused, since its codes depend on the process that
made them; cast it to a ``pl.Enum``. The label and the weight are numeric,
with no nulls and finite values (a weight not negative).

Requires the library itself, which is not a dependency of this package
(``pip install xgboost`` or ``lightgbm``), and numpy, which both libraries
require.

.. warning::
    This module is considered **unstable**: it is built on the libraries' C
    APIs and on how their Python classes hold a native handle, so its
    keywords and what it supports may change in any release without that
    counting as a breaking change, and a new library version may need a new
    release here. Under ``POLARS_ONLINE_WARN_UNSTABLE=1`` each call raises
    :class:`polars_online.UnstableWarning`.
"""

from __future__ import annotations

import ctypes
import functools
import glob
import os
import warnings
import weakref
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from polars_online import _polars_online as _native
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


def _format(path: str | os.PathLike[str]) -> str:
    ext = Path(os.fspath(path)).suffix.lower()
    if ext in (".arrow", ".ipc", ".feather"):
        return "ipc"
    if ext in (".parquet", ".pq"):
        return "parquet"
    raise ValueError(
        f"cannot tell the format of {os.fspath(path)!r} from its extension: po.gbdt reads Arrow "
        "IPC (.arrow, .ipc, .feather) and Parquet (.parquet, .pq) files, or a LazyFrame"
    )


def _expand(paths: list[str | os.PathLike[str]]) -> list[str]:
    """The files a list of paths and globs names, each glob's sorted, as
    polars' scans order them."""
    out: list[str] = []
    for p in paths:
        s = os.fspath(p)
        if any(c in s for c in "*?["):
            found = sorted(glob.glob(s))
            if not found:
                raise FileNotFoundError(f"no file matches {s!r}")
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
            f"{who}: source must be paths, a LazyFrame or a DataFrame, not {type(source).__name__}"
        )
    if not paths:
        raise ValueError(f"{who}: source names no files")
    formats = {_format(p) for p in paths}
    if len(formats) > 1:
        raise ValueError(f"{who}: source mixes Arrow IPC and Parquet files; pass one kind")
    names = [os.fspath(p) for p in paths]
    if formats.pop() == "ipc":
        return pl.scan_ipc(names), _expand(paths)
    return pl.scan_parquet(names), None


class _Columns:
    """The features, label and weight of a source, checked, and the plan
    that selects and casts them as both engines read them: a ``pl.Enum`` as
    its codes, a Boolean as ``UInt8``, the label and weight as ``Float64``."""

    def __init__(
        self,
        plan: pl.LazyFrame,
        *,
        label: str,
        weight: str | None,
        features: Sequence[str] | None,
        who: str,
    ) -> None:
        schema = plan.collect_schema()
        for role, name in (("label", label), ("weight", weight)):
            if name is not None and name not in schema:
                raise ValueError(f"{who}: the {role} column {name!r} is not in the source")
        if features is None:
            names = [c for c in schema if c not in (label, weight)]
        else:
            if isinstance(features, str):
                raise TypeError(f"{who}: features is a list of column names, not one name")
            names = list(features)
            missing = [c for c in names if c not in schema]
            if missing:
                raise ValueError(f"{who}: feature columns not in the source: {missing}")
            clash = [c for c in names if c in (label, weight)]
            if clash:
                raise ValueError(f"{who}: {clash} named as a feature and as the label or weight")
            if len(set(names)) != len(names):
                raise ValueError(f"{who}: a feature is named twice in {names}")
        if not names:
            raise ValueError(f"{who}: the source has no feature columns besides the label")
        self.names = names
        self.label = label
        self.weight = weight
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
                raise ValueError(
                    f"{who}: feature {c!r} is a pl.Categorical, whose codes depend on the "
                    "process that made them, so a model trained on them could not be read "
                    "again; cast it to a pl.Enum, whose categories fix the codes"
                )
            else:
                raise ValueError(
                    f"{who}: feature {c!r} is {dt}: po.gbdt reads numeric, Boolean and pl.Enum "
                    "columns; cast it to one of them or leave it out of features"
                )
            if len(self.categories) < len(self.kinds):
                self.categories.append(None)
        for role, name in (("label", label), ("weight", weight)):
            if name is None:
                continue
            dt = schema[name]
            if not (dt.is_numeric() or dt == pl.Boolean) or dt.is_decimal():
                raise ValueError(f"{who}: the {role} column {name!r} is {dt}, not numeric")
            exprs.append(pl.col(name).cast(pl.Float64))
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
                    f"the {role} column `{name}` has a null at row {at}: every row needs a {role}"
                )
            bad = ~s.is_finite() if role == "label" else (~s.is_finite()) | (s < 0)
            if bad.any():
                i = int(bad.arg_max() or 0)
                need = "finite" if role == "label" else "finite and not negative"
                raise ValueError(
                    f"the {role} column `{name}` holds {s[i]} at row {row + i}: a {role} "
                    f"must be {need}"
                )


def _check_common(who: str, max_bin: int, threads: int | None, engine: str) -> None:
    if isinstance(max_bin, bool) or not isinstance(max_bin, int) or max_bin < 2:
        raise ValueError(f"{who}: max_bin must be an integer of 2 or more, not {max_bin!r}")
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError(f"{who}: threads must be a positive integer or None, not {threads!r}")
    if engine not in _ENGINES:
        raise ValueError(f"{who}: engine must be one of {_ENGINES}, not {engine!r}")


def _library_path(lib: Any) -> str:
    """The file a library's Python package loaded it from."""
    path = getattr(lib, "path", None) or getattr(lib, "_name", None)
    if not path:
        raise ValueError("the library's Python package does not say which file it loaded")
    return str(path)


@functools.cache
def _xgb_native(path: str) -> Any:
    return _native.GbdtXgboost(path)


@functools.cache
def _lgb_native(path: str, version: str) -> Any:
    return _native.GbdtLightgbm(path, version)


def _native_or_none(load: Callable[[], Any], engine: str, what: str, stacklevel: int) -> Any:
    if engine == "python":
        return None
    try:
        return load()
    except (ValueError, OSError) as e:
        if engine == "native":
            raise
        warnings.warn(
            f"po.gbdt cannot load {what} natively ({e}); building with the Python engine, which "
            "grows the same trees more slowly. Pass engine='python' to choose it and silence this.",
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
    threads: int | None = None,
    seed: int = 0,
    engine: Engine = "auto",
) -> xgboost.QuantileDMatrix:
    """An ``xgboost.QuantileDMatrix`` of ``source``, binned in one stream.

    XGBoost places each feature's bins at its quantiles and reads the input
    in passes: it counts the values, sketches the quantiles over every row,
    indexes each value into its bin and builds the column matrix. The
    returned matrix is XGBoost's own, and ``xgb.train`` takes it as it is.
    Here from Arrow IPC files, which the native engine reads itself:

    .. code-block:: python

        import xgboost as xgb

        df.head(300).write_ipc("train.arrow")              # the example data, as two Arrow files
        df.tail(100).write_ipc("valid.arrow")
        features = ["x0", "x1", "x2", "signal_a", "signal_b"]
        train = po.gbdt.xgboost_matrix("train.arrow", label="y", features=features)
        valid = po.gbdt.xgboost_matrix("valid.arrow", label="y", features=features, ref=train)
        booster = xgb.train({"tree_method": "hist", "max_bin": 256}, train, 500,
                            evals=[(valid, "valid")], early_stopping_rounds=20, verbose_eval=False)

    **The cuts.** With ``sample_rows``, one pass first draws a uniform sample
    of that many rows (row ``i`` is kept when its hash, SplitMix64 of ``i``
    and ``seed``, is among the ``sample_rows`` smallest) and places each
    feature's bin edges at the sample's quantiles: a feature with at most
    ``max_bin`` distinct sample values gets a bin per value, and any other
    gets ``max_bin`` bins of equal sample frequency, edge ``j`` at the sorted
    sample's value of rank ``floor(j m / max_bin)`` for ``m`` values. They
    reach XGBoost as a ``ref``, which removes its sketch pass. The sample is
    the same rows however the source is batched and whichever engine reads
    it. ``sample_rows=None`` lets XGBoost sketch every row, and grows
    XGBoost's own trees.

    **Measured** on 10 million rows of 50 features with nulls, from Arrow
    IPC files, at 8 threads (``docs/GBDT-ADAPTER.md`` §7): the first round
    began after 4.0 seconds at a peak of 3.6 GB, against 8.7 s and 4.8 GB for
    XGBoost's own ``DataIter`` over Polars batches, and 7.0 s and 8.7 GB for
    ``collect()`` and numpy. ``sample_rows=None`` took 8.0 s. A
    ``LazyFrame`` took 4.2 s, at the 7.2 GB that Polars' read-ahead holds
    (the README's "Tuning memory with Polars' own settings" cuts it), and
    ``engine="python"`` 5.5 s.

    Parameters
    ----------
    source
        Arrow IPC or Parquet paths (globs taken, one kind of file), a
        ``LazyFrame``, read in batches through ``collect_batches`` once per
        pass and so re-run each time, or a ``DataFrame``. The native engine
        reads Arrow IPC files itself, a record batch at a time.
    label, weight
        The label and the row weights: numeric columns with no nulls and
        finite values, a weight not negative.
    features
        The feature columns, in order; every other column by default.
    max_bin
        Bins per feature, XGBoost's ``max_bin``: training must pass the same.
    sample_rows
        Rows of the sample the cuts come from; ``None`` for XGBoost's own
        sketch. Ignored with ``ref``.
    ref
        The training matrix, for a validation or test matrix binned on its
        cuts (``xgb.train`` refuses a ``QuantileDMatrix`` evaluation set
        without it). The returned matrix keeps a weak reference to it.
    threads
        Threads for XGBoost's construction and the conversions; all by
        default.
    seed
        The sample's seed.
    engine
        ``"native"``, ``"python"``, or ``"auto"``: native, falling back to
        Python with a warning when the native engine cannot load XGBoost.

    Raises
    ------
    ValueError
        Each named: ``max_bin`` below 2, ``threads`` or ``sample_rows`` not
        a positive integer, or an ``engine`` not one of the three; a source
        that names no files, mixes Arrow IPC and Parquet, or names a file
        whose extension says neither; a column missing, named twice, or of
        a type the libraries do not read (a ``pl.Categorical`` among them);
        a null or non-finite label or weight, or a negative weight; an empty
        source, or one whose passes give different rows; a ``ref`` binned on
        other features; or (``engine="native"``) an XGBoost the native
        engine does not support.
    TypeError
        ``ref`` not an ``xgboost.DMatrix``, a source of another type, or
        ``features`` given as one name.
    FileNotFoundError
        A glob of Arrow IPC files that matches nothing.
    """
    import xgboost as xgb

    who = "xgboost_matrix"
    _check_common(who, max_bin, threads, engine)
    if sample_rows is not None and (
        isinstance(sample_rows, bool) or not isinstance(sample_rows, int) or sample_rows < 1
    ):
        raise ValueError(
            f"{who}: sample_rows must be a positive integer or None, not {sample_rows!r}"
        )
    if ref is not None and not isinstance(ref, xgb.DMatrix):
        raise TypeError(f"{who}: ref must be an xgboost.DMatrix, not {type(ref).__name__}")
    plan, files = _plan(source, who)
    cols = _Columns(plan, label=label, weight=weight, features=features, who=who)
    if ref is not None and ref.feature_names is not None and list(ref.feature_names) != cols.names:
        raise ValueError(
            f"{who}: ref has the features {list(ref.feature_names)}, and this source "
            f"{cols.names}: a matrix binned on another's cuts needs the same features in order"
        )
    native = _native_or_none(
        lambda: _xgb_native(_library_path(xgb.core._LIB)), engine, "XGBoost", 4
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
        )
        _LAST_STATS.clear()
        _LAST_STATS.update(stats)
        # The constructor's branch for a handle made elsewhere: the matrix
        # owns it from here and frees it once, when it is collected.
        out = xgb.QuantileDMatrix(ctypes.c_void_p(handle), max_bin=max_bin, nthread=threads)
    else:
        out = _xgb_python(xgb, cols, max_bin, sample_rows, ref, threads, seed)
    if ref is not None:
        out.ref = weakref.ref(ref)
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
        )
        for frame in cols.batches():
            sampler.push(frame)
        if sampler.rows_seen == 0:
            raise ValueError("the source has no rows")
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
                    raise ValueError("the source has no rows")
                if self._rows is not None and self._rows != self._row:
                    raise ValueError(
                        f"the source gave {self._rows} rows in one pass and {self._row} in "
                        "another: XGBoost reads it more than once, and every pass must give "
                        "the same rows"
                    )
                self._rows = self._row
                return False
            if frame.height == 0:
                return self.next(input_data)
            cols.check_meta(frame, self._row)
            meta: dict[str, Any] = {"label": frame[cols.label].to_numpy()}
            if cols.weight is not None:
                meta["weight"] = frame[cols.weight].to_numpy()
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
    in_memory: bool | Literal["auto"] = "auto",
    threads: int | None = None,
    engine: Engine = "auto",
) -> lightgbm.Dataset:
    """A constructed ``lightgbm.Dataset`` of ``source``, binned on
    LightGBM's own sample.

    LightGBM places each feature's bins from a sample of
    ``bin_construct_sample_cnt`` rows (200,000 by default), drawn from the
    row count and ``data_random_seed``, and then pushes every value into its
    bin. Here the source is counted (``select(pl.len())``), a first pass
    gathers the rows LightGBM samples, and a second pushes every row as
    float32, built in parallel: two passes, and the data never held. With
    ``in_memory``, one pass holds the features as float32 Arrow and hands
    LightGBM the whole stream, which it pushes column by column. Both routes
    hand LightGBM the same float32 values and grow the same trees, which are
    LightGBM's own from a float32 array of the data. The returned dataset is
    LightGBM's own, and ``lgb.train`` takes it as it is. Here the training
    rows are the example data's two parquet files, read by a glob:

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

    The dataset is constructed and its raw data never kept, so training may
    not change a dataset parameter (``max_bin``, ``min_data_in_bin``,
    ``data_random_seed``, ...): LightGBM refuses, as it does for any
    constructed dataset. Pass them here, in ``params``.

    **Measured** on 10 million rows of 50 features with nulls, from Arrow
    IPC files, at 8 threads (``docs/GBDT-ADAPTER.md`` §7): the first round
    began after 1.7 seconds at a peak of 1.7 GB streamed, and after 1.45 s at
    2.9 GB in memory, against 1.5 s and 8.9 GB for LightGBM's own route from
    a collected Polars frame, and 2.2 s for its ``Sequence``.

    Parameters
    ----------
    source
        Arrow IPC or Parquet paths (globs taken, one kind of file), a
        ``LazyFrame`` (re-run for each pass), or a ``DataFrame``. The native
        engine reads Arrow IPC files itself, a record batch at a time.
    label, weight
        The label and the row weights: numeric columns with no nulls and
        finite values, a weight not negative.
    features
        The feature columns, in order; every other column by default.
    categorical
        The categorical features, ``pl.Enum`` or integer columns (a code
        below 0 is missing, as LightGBM reads it); every ``pl.Enum`` feature
        by default. A ``pl.Enum`` left out goes in as its codes, a number.
    max_bin
        Bins per feature, LightGBM's ``max_bin``.
    params
        Other dataset parameters, as ``lightgbm.Dataset(params=)`` takes
        them: ``verbose``, ``min_data_in_bin``, ``bin_construct_sample_cnt``,
        ``data_random_seed``, ``use_missing``, ``zero_as_missing``, ... Not
        ``max_bin``, ``categorical_feature`` or ``num_threads``, which are
        this function's own arguments.
    reference
        The training dataset, for a validation dataset binned on its bins;
        one pass.
    in_memory
        Hand LightGBM the whole stream at once, held in memory: as Arrow
        under the native engine, as a float32 numpy array under the Python
        one. ``"auto"`` does when the features as float32 take at most a
        quarter of the memory available and, under the native engine, the
        LightGBM loaded takes an Arrow stream (4.7.0 on).
    threads
        Threads for LightGBM's construction and the conversions; all by
        default.
    engine
        ``"native"``, ``"python"``, or ``"auto"``: native, falling back to
        Python with a warning when the native engine cannot load LightGBM.

    Raises
    ------
    ValueError
        Each named: ``max_bin`` below 2, ``threads`` not a positive integer,
        an ``engine`` not one of the three, or an ``in_memory`` that is not
        ``True``, ``False`` or ``"auto"``; a parameter in ``params`` that is
        this function's own argument; a source that names no files, mixes
        Arrow IPC and Parquet, or names a file whose extension says neither;
        a column missing, named twice, or of a type the libraries do not
        read (a ``pl.Categorical`` among them); a ``categorical`` name that
        is not a feature, or is a float column; a null or non-finite label
        or weight, or a negative weight; an empty source, more rows than
        LightGBM's 32-bit row count holds, or a source whose passes give
        different rows; or (``engine="native"``) a LightGBM the native
        engine does not support.
    TypeError
        ``reference`` not a ``lightgbm.Dataset`` (one not yet constructed is
        constructed here), a source of another type, or ``features`` or
        ``categorical`` given as one name.
    FileNotFoundError
        A glob of Arrow IPC files that matches nothing.
    """
    import lightgbm as lgb

    who = "lightgbm_dataset"
    _check_common(who, max_bin, threads, engine)
    if in_memory not in (True, False, "auto"):
        raise ValueError(f"{who}: in_memory must be True, False or 'auto', not {in_memory!r}")
    if reference is not None:
        if not isinstance(reference, lgb.Dataset):
            raise TypeError(
                f"{who}: reference must be a lightgbm.Dataset, not {type(reference).__name__}"
            )
        reference.construct()
    given = dict(params or {})
    for key, arg in _LGB_OWN.items():
        if key in given:
            raise ValueError(f"{who}: params names {key!r}; pass it as {arg}= instead")
    plan, files = _plan(source, who)
    cols = _Columns(plan, label=label, weight=weight, features=features, who=who)
    if categorical is None:
        cats = [j for j, k in enumerate(cols.kinds) if k == "category"]
    else:
        if isinstance(categorical, str):
            raise TypeError(f"{who}: categorical is a list of column names, not one name")
        cats = []
        for c in categorical:
            if c not in cols.names:
                raise ValueError(f"{who}: categorical names {c!r}, which is not a feature")
            j = cols.names.index(c)
            if cols.kinds[j] not in ("category", "int", "bool"):
                raise ValueError(
                    f"{who}: categorical names {c!r}, a float column: LightGBM reads a "
                    "categorical feature as integer codes"
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
        raise ValueError(f"{who}: the source has no rows")
    native = _native_or_none(
        lambda: _lgb_native(_library_path(lgb.basic._LIB), lgb.__version__),
        engine,
        "LightGBM",
        4,
    )
    if in_memory == "auto":
        fits = rows * len(cols.names) * 4 * 4 <= _available_memory()
        memory = fits and (native is None or native.has_arrow)
    else:
        memory = bool(in_memory)
    if native is not None:
        if not hasattr(lgb.Dataset(None), "_handle"):
            raise ValueError(
                f"{who}: this lightgbm.Dataset holds its native handle elsewhere than "
                f"`_handle` (lightgbm {lgb.__version__}); engine='python' builds through its "
                "Python API"
            )
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
        )
        _LAST_STATS.clear()
        _LAST_STATS.update(stats)
        ds = lgb.Dataset(None, reference=reference, params=dparams, free_raw_data=True)
        # The dataset owns the handle from here and frees it once
        # (`Dataset._free_handle`, on collection).
        ds._handle = ctypes.c_void_p(handle)
        ds.feature_name = list(cols.names)
        ds.categorical_feature = cat_attr
        return ds
    return _lgb_python(lgb, cols, rows, dparams, build, cat_attr, reference, memory)


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
    as_f32 = [pl.col(c).cast(pl.Float32) for c in cols.names]

    def meta(frame: pl.DataFrame, row: int) -> None:
        cols.check_meta(frame, row)
        labels.append(frame[cols.label].to_numpy())
        if cols.weight is not None:
            weights.append(frame[cols.weight].to_numpy())

    if memory:
        frames = []
        row = 0
        for frame in cols.batches():
            meta(frame, row)
            frames.append(frame.select(as_f32))
            row += frame.height
        if row != rows:
            raise ValueError(
                f"the source was counted at {rows} rows and a pass over it gave {row}: "
                "LightGBM reads it more than once, and every pass must give the same rows"
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
                f"the source was counted at {rows} rows and a pass over it gave {row}: "
                "LightGBM reads it more than once, and every pass must give the same rows"
            )
        data = _lgb_rows(lgb, cols, rows, gathered, as_f32)
    label = np.concatenate(labels).astype(np.float32)
    weight = np.concatenate(weights).astype(np.float32) if cols.weight is not None else None
    ds = lgb.Dataset(
        data,
        label=label,
        weight=weight,
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
                raise RuntimeError("po.gbdt's Sequence is read in order")
            if self._it is None:
                self._it = cols.batches()
            while self._buf.shape[0] < stop - start:
                frame = next(self._it, None)
                if frame is None:
                    raise ValueError(
                        f"the source was counted at {rows} rows and a pass over it gave "
                        f"{self._at + self._buf.shape[0]}: LightGBM reads it more than once, "
                        "and every pass must give the same rows"
                    )
                block = frame.select(as_f32).to_numpy(order="c")
                self._buf = np.concatenate([self._buf, block]) if self._buf.size else block
            out, self._buf = self._buf[: stop - start], self._buf[stop - start :]
            self._at = stop
            return out

    return Rows()
