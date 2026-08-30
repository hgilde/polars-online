"""Streaming parquet runner (ENHANCEMENTS E8).

The same code path the ``online`` CLI uses, callable from Python: memory stays
O(state + chunk) rather than O(data), without spawning a process.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from polars_online import _polars_online as _native
from polars_online._spec import _json

__all__ = ["run"]


def run(
    config: dict[str, Any] | str | Path | None = None,
    *,
    input: str | Path | None = None,  # noqa: A002 - mirrors the TOML key
    output: str | Path | None = None,
    specs: Iterable[dict[str, Any]] | None = None,
    chunk_rows: int | None = None,
    load_state: str | Path | None = None,
    save_state: str | Path | None = None,
) -> dict[str, int]:
    """Stream a parquet file through a model bank, writing parquet out.

    ``config`` is a dict, a path to a TOML file, or ``None`` to build the config
    from the keyword arguments. Keywords override whatever the config supplies,
    so a checked-in TOML can be reused with a different input::

        po.run("bank.toml", input="today.parquet", output="today-out.parquet")

    Returns ``{"rows": ..., "chunks": ...}``. Chunking never changes the
    numbers -- it only trades memory for overhead -- so ``chunk_rows`` is purely
    a resource knob.
    """
    if isinstance(config, (str, Path)):
        cfg = tomllib.loads(Path(config).read_text())
    elif config is None:
        cfg = {}
    else:
        cfg = dict(config)

    overrides = {
        "input": input,
        "output": output,
        "specs": list(specs) if specs is not None else None,
        "chunk_rows": chunk_rows,
        "load_state": load_state,
        "save_state": save_state,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    for key in ("input", "output", "load_state", "save_state"):
        if cfg.get(key) is not None:
            cfg[key] = str(cfg[key])
    if not cfg.get("specs"):
        msg = "run() needs at least one spec, from `specs=` or the config's [[specs]]"
        raise ValueError(msg)

    rows, chunks = _native.run_config(_json(cfg))
    return {"rows": rows, "chunks": chunks}
