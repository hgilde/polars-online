# polars-online

Streaming / online regression models for Polars. Rust core, exposed as
(1) a Polars expression plugin, (2) a chunk-fed Python "model bank",
(3) a standalone Rust CLI for deployment. Runs on data that does not fit in memory.

**Read `docs/PLAN.md` before doing anything.** It is the source of truth for the
design and the task list. Tick tasks off there as they are completed; add
decisions there, not in chat.

## Stack

- Rust (stable), Cargo workspace. Linear algebra: `faer`. Serialization: `serde` + `rmp-serde`.
- Python bindings: `pyo3` + `pyo3-polars`, built with `maturin`.
- Python: 3.12+, managed with `uv`. Polars version pinned in `pyproject.toml`
  and `Cargo.toml` — they must match.
- Dev on macOS (arm64); deploy on macOS and Windows. Never use platform-specific code paths
  without a cfg-guarded fallback.

## Layout

```
crates/online-core/    pure Rust models, NO polars dependency, exhaustively unit-tested
crates/online-polars/  Rust-side integration: model bank over Polars DataFrames / parquet streams
crates/online-cli/     binary: parquet in -> parquet out, config from TOML
crates/online-py/      pyo3 + pyo3-polars: expression plugin + Python ModelBank class
python/polars_online/  Python package (thin wrappers, expression namespace registration)
tests/                 pytest (Python) — integration, invariance, oracle tests
docs/PLAN.md           design + task list
```

## Commands

```
uv sync                                  # Python env
cargo test --workspace                   # Rust unit tests
maturin develop --release -m crates/online-py/Cargo.toml
uv run pytest -x                         # Python tests (downloads/generates data on first run)
cargo run -p online-cli -- --config examples/bank.toml
```

## Hard rules

1. **Tests download or generate their own data.** No data files in the repo, ever.
   Use `tests/data.py` (seeded synthetic generator + cached public download). Downloaded
   data is cached under `.cache/` (gitignored); tests needing it are skipped when offline.
2. **Predictions are out-of-sample by construction**: `pred` is computed before the update
   with the current row's target. Any test that finds otherwise is a bug, not a flake.
3. **Chunk invariance**: feeding a stream as 1 chunk or 1000 chunks must give identical
   output. There is a test for this; keep it passing.
4. Models live in `online-core` behind the `OnlineModel` trait and know nothing about
   Polars, Python, or clocks-as-columns. Plumbing lives in `online-polars` / `online-py`.
5. State files are versioned msgpack and must load on both OSes; bump `SCHEMA_VERSION`
   on any layout change and keep a loader for the previous version.
6. No `unsafe` in `online-core`. f64 everywhere.
7. Commit after each completed task in `docs/PLAN.md`, with the task number in the message.

## Style

- Rust: `cargo fmt`, `cargo clippy -D warnings`. Small files, one model per file.
- Python: `ruff` (format + lint), type hints, no pandas.
- Docstrings state the math (update equations) for every model.
