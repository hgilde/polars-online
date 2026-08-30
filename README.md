# polars-online

Streaming / online regression models for Polars. Rust core, exposed as a Polars
expression plugin, a chunk-fed Python model bank, and a standalone Rust CLI.

Design and task list: [`docs/PLAN.md`](docs/PLAN.md).

## Development

```sh
uv sync                                                # Python env (CPython 3.12)
uv run cargo test --workspace                          # Rust unit tests
uv run maturin develop --release -m crates/online-py/Cargo.toml
uv run pytest                                          # Python tests
```

`cargo` is run via `uv run` because `online-py` builds against pyo3's `abi3-py312`,
which needs a >= 3.12 interpreter at build time; `uv run` exports `VIRTUAL_ENV`, which
pyo3's build script picks up. Plain `cargo test` also works if `PYO3_PYTHON` points at
a 3.12+ interpreter.

## Version pins

`polars` is pinned in two places that must stay in sync
(`Cargo.toml` and `pyproject.toml`):

| py-polars | rust polars | pyo3-polars | pyo3 |
|---|---|---|---|
| 1.44.1 | 0.55.2 | 0.28 | 0.29 |
