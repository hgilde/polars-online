"""Streaming runner (ENHANCEMENTS E8, E32).

The same pipeline the ``online`` CLI runs, callable from Python: polars reads
the source in chunks, the bank fits and predicts, and a writer thread writes
the augmented frames out -- one chunk in flight per stage, so memory stays
O(state + chunk) rather than O(data), without spawning a process. The reading
here is py-polars' own (``LazyFrame.collect_batches``), so any source polars
can scan -- a path in any format, a glob, a cloud URL, a query -- streams
through, and so does any iterable of frames.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import polars as pl

from polars_online import _polars_online as _native
from polars_online._spec import _json

__all__ = ["run"]

Source = str | os.PathLike[str] | pl.LazyFrame | pl.DataFrame | Iterable[pl.DataFrame]

_SCAN: dict[str, Callable[[str], pl.LazyFrame]] = {
    "parquet": pl.scan_parquet,
    "ipc": pl.scan_ipc,
    "csv": pl.scan_csv,
    "ndjson": pl.scan_ndjson,
}


def run(
    config: dict[str, Any] | str | Path | None = None,
    *,
    input: Source | None = None,  # noqa: A002 - mirrors the TOML key
    output: str | os.PathLike[str] | None = None,
    specs: Iterable[dict[str, Any]] | None = None,
    chunk_rows: int | None = None,
    load_state: str | os.PathLike[str] | None = None,
    save_state: str | os.PathLike[str] | None = None,
    predict: bool | None = None,
    input_format: str | None = None,
    output_format: str | None = None,
    keep_columns: Iterable[str] | None = None,
    progress: Callable[[int, int], object] | None = None,
) -> dict[str, int]:
    """Stream rows through a model bank and write them out with its columns.

    ``input`` is a path (parquet, ipc, csv or ndjson, told from the extension
    or named by ``input_format``; globs and cloud URLs as ``pl.scan_*`` takes
    them), a ``LazyFrame`` (any query: the scan is polars', with whatever
    options it needs), a ``DataFrame``, or any iterable of ``DataFrame``\\ s in
    stream order -- chunks from a database cursor, a socket, a generator.
    ``output`` is a path in any of the four formats, told the same way; it is
    written through a temporary and renamed into place, so a run that fails
    leaves the previous file where it was. CSV cannot hold the bank's struct
    columns, so there each spec's struct is flattened to ``<spec>.<field>``
    columns and a list field (``coef``) becomes a JSON string --
    ``pl.col("ridge.coef").str.json_decode(pl.List(pl.Float64))`` reads it
    back.

    ``config`` is a dict, a path to a TOML file, or ``None`` to build the
    config from the keyword arguments. Keywords override whatever the config
    supplies, so a checked-in TOML can be reused with a different input::

        po.run("bank.toml", input="today.csv", output="today-out.parquet")

    Returns ``{"rows": ..., "chunks": ...}``. Chunking never changes the
    numbers -- it only trades memory for overhead -- so ``chunk_rows`` (the
    reader's chunk; frames passed in directly are taken as they come) is
    purely a resource knob. On data sorted by group, a chunk should span
    several groups: the bank fits groups in parallel within a chunk.

    ``keep_columns`` selects input columns before the bank sees them (and
    before the scan reads them). ``progress(rows, chunks)`` is called after
    each chunk; raising in it stops the run without publishing the output.

    ``predict=True`` scores instead of learning: every row gets what the bank
    loaded from ``load_state`` predicts for it as it stands
    (:meth:`ModelBank.predict`), and the bank is not updated -- so it needs
    ``load_state`` and refuses ``save_state``. One TOML can serve both runs:
    the keyword drops the config's ``save_state``, which belongs to the
    learning run, unless ``save_state=`` is passed alongside it.

    What is wrong with the call or the config is ``ValueError``, before a
    row is read: no input; no specs; a spec the bank refuses
    (:class:`ModelBank`); a key the config, a spec or its model has not
    got, named with the keys there are (a misspelt key is never kept at its
    default in silence); ``chunk_rows`` below 1; a format that cannot be
    told from a path's extension, or that is not one of the four;
    ``predict=True`` without ``load_state``, or with ``save_state``; an
    iterable that produced no frames; a ``load_state`` that is not a bank
    this build loads or whose specs are not ``specs``; and a TOML that does
    not parse (``tomllib.TOMLDecodeError``). ``TypeError`` for a ``config``
    that is none of the three, a ``progress`` that is not callable, or an
    item of ``input`` that is not a ``DataFrame``. A file fails as the
    ``OSError`` for what went wrong, with the path in the message: a
    ``config`` or ``input`` that is not there (the scan is polars', so its
    ``FileNotFoundError``), a ``load_state`` that cannot be read, an
    ``output`` whose directory is not there, and a ``save_state`` whose
    directory is not -- found out before the run, since after it the output
    would be written and the state lost. A column the specs read that the
    input has not got, or that ``keep_columns`` dropped, is the bank's
    ``ValueError`` (a ``keep_columns`` name the input has not got is
    polars' ``ColumnNotFoundError``); a value the bank refuses -- a null
    clock, a negative weight, a clock running backwards -- is its
    ``ValueError`` mid-run. Whatever stops the run -- the bank, the writer,
    ``progress`` or the iterable raising (both come through as themselves)
    -- leaves the previous ``output`` where it was and ``save_state``
    unwritten: the state is saved last, after the output is in place, so a
    state file always has an output to go with it.
    """
    if isinstance(config, (str, Path)):
        cfg = tomllib.loads(Path(config).read_text())
    elif config is None:
        cfg = {}
    elif isinstance(config, dict):
        cfg = dict(config)
    else:
        msg = f"config must be a dict, a path to a TOML file, or None, got {type(config).__name__}"
        raise TypeError(msg)

    overrides: dict[str, Any] = {
        "output": output,
        "specs": list(specs) if specs is not None else None,
        "chunk_rows": chunk_rows,
        "load_state": load_state,
        "save_state": save_state,
        "predict": predict,
        "input_format": input_format,
        "output_format": output_format,
        "keep_columns": list(keep_columns) if keep_columns is not None else None,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    if predict and save_state is None:
        cfg.pop("save_state", None)

    # The source is read here, by py-polars, and only its frames cross into
    # Rust; the config's `input` is a path the TOML may carry.
    source: Source | None = input if input is not None else cfg.pop("input", None)
    if source is None:
        msg = "run() needs an input: a path, a LazyFrame, a DataFrame, or an iterable of DataFrames"
        raise ValueError(msg)
    for key in ("output", "load_state", "save_state"):
        if cfg.get(key) is not None:
            cfg[key] = os.fspath(cfg[key])
    if not cfg.get("specs"):
        msg = "run() needs at least one spec, from `specs=` or the config's [[specs]]"
        raise ValueError(msg)
    if progress is not None and not callable(progress):
        msg = f"progress must be callable, got {type(progress).__name__}"
        raise TypeError(msg)
    cfg.setdefault("chunk_rows", _native.default_chunk_rows())
    if not isinstance(cfg["chunk_rows"], int) or cfg["chunk_rows"] < 1:
        msg = f"chunk_rows must be at least 1, got {cfg['chunk_rows']!r}"
        raise ValueError(msg)

    frames, schema = _frames(source, cfg)
    rows, chunks = _native.run_config_frames(_json(cfg), frames, schema, progress)
    return {"rows": rows, "chunks": chunks}


def _frames(source: Source, cfg: dict[str, Any]) -> tuple[Iterator[pl.DataFrame], pl.DataFrame]:
    """The source as an iterator of frames in stream order, plus an empty
    frame with their schema (the output's, when there are no frames).

    A path or a plan is read by polars' streaming engine in ``chunk_rows``
    chunks, with ``keep_columns`` pushed into the plan so the scan reads only
    those columns; frames handed in directly are taken as they come, and the
    runner applies ``keep_columns`` to each."""
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        fmt = cfg.get("input_format") or _native.format_of_path(path)
        if fmt not in _SCAN:
            msg = f"input_format {fmt!r} is not one of {', '.join(_native.formats())}"
            raise ValueError(msg)
        lf = _SCAN[fmt](path)
    elif isinstance(source, pl.DataFrame):
        lf = source.lazy()
    elif isinstance(source, pl.LazyFrame):
        lf = source
    else:
        it = iter(source)
        first = next(it, None)
        if first is None:
            msg = (
                "input produced no frames; a stream with no rows still needs a schema, "
                "so pass it as a DataFrame or LazyFrame"
            )
            raise ValueError(msg)
        if not isinstance(first, pl.DataFrame):
            raise TypeError(_not_a_frame(first))
        return _checked([first], it), first.clear()

    if cfg.get("keep_columns"):
        lf = lf.select(cfg["keep_columns"])
    schema = pl.DataFrame(schema=lf.collect_schema())
    batches = lf.collect_batches(chunk_size=cfg["chunk_rows"], maintain_order=True)
    return iter(batches), schema


def _checked(*parts: Iterable[object]) -> Iterator[pl.DataFrame]:
    """`parts` chained, each item held to be a DataFrame."""
    for part in parts:
        for item in part:
            if not isinstance(item, pl.DataFrame):
                raise TypeError(_not_a_frame(item))
            yield item


def _not_a_frame(item: object) -> str:
    return f"input frames must be polars DataFrames, got {type(item).__name__}"
