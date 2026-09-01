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
8. **`n_eff` means the same thing in every model**: the accumulated weight *before* this
   row's update and *before* its own decay. That is what makes `min_periods` portable
   across a bank. `sgd` and `pa` once applied the row's decay first, so `min_periods`
   quietly meant a different number of rows for them; `crates/online-core/tests/model_contract.rs`
   now checks every model against the same recursion.
9. **A zero-weight row is legal** and means "advance the clock, learn nothing" — including
   as the *first* row of a stream, where `lam*w_sum + w` is 0 and the mean-form update's
   `a` and `b` are both 0/0. Guard every such division; an unguarded one poisons the state
   with a NaN that never washes out.

## Linking

10. **Use the Rust API the way Rust intends, and check the docs rather than
    recalling them.** The Reference is installed locally:
    `$(rustc --print sysroot)/share/doc/rust/html/reference/linkage.html`.
11. **Prefer the Rust-native way to connect Rust components.** That is
    `crate-type = "dylib"` plus `-C prefer-dynamic`, not a hand-rolled C shim.
    The compiler consumes a dependency in exactly two forms — `rlib` or
    `dylib` — and which are available is the *publishing* crate's choice.
12. **Do not add static linking of anything new without raising it first.**
    If a change would statically link a library that is not already linked,
    stop and ask. This includes vendoring a C library through a `-sys` crate.

**Standing exception, already raised and unresolved (2026-08-31).** Everything
in this workspace is statically linked today, and it is not a choice that was
made:

- A Python extension module must be a `cdylib` — the Reference: "used when
  compiling a dynamic library to be loaded from another language" — and a
  `cdylib` statically links its Rust dependency graph by construction.
- **Zero of our 453 dependencies publishes a `dylib` target.** 419 are plain
  `lib` (rlib). Not polars, not faer, not serde, not pyo3. `crate-type` is
  declared by the crate being published, so this cannot be overridden
  downstream, and near-nothing on crates.io publishes a dylib.
- py-polars exports **no Rust symbols at all** — 33 on macOS, 32 on Linux, none
  mangled — so there is nothing to bind to even in principle, and Rust has no
  stable ABI to bind with.

The one part under our control was tested rather than assumed: building
`online-core` and `online-polars` as dylibs and adding `-C prefer-dynamic`
*does* work, and the extension then links `libonline_core.dylib` and
`libonline_polars.dylib` dynamically. It was **not** adopted, and the reasons
belong with the rule so nobody re-litigates it from scratch:

- It also drags in `libstd-<hash>.dylib`, a **toolchain-version-pinned** copy of
  the Rust standard library that exists only inside a rustup install. Shipping
  that in a wheel means shipping libstd and matching the exact rustc forever.
- It saves nothing: polars stays statically inside whichever dylib uses it,
  because polars is rlib-only. The bytes move, they do not shrink.
- It multiplies the artifacts a wheel must carry and the rpaths it must wire,
  per platform, for components that are always versioned and shipped together.

Revisit if polars ever ships a `dylib` target. Until then, "do not statically
link" is achievable for *new* dependencies only, which is what rule 12 asks.

## Style

- Rust: `cargo fmt`, `cargo clippy -D warnings`. Small files, one model per file.
- Python: `ruff` (format + lint), type hints, no pandas.
- Docstrings state the math (update equations) for every model.
