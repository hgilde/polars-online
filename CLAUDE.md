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

**Raised and resolved (2026-08-31): static linking here is what Polars
prescribes, not a shortcut.** Checked against Polars' own documentation rather
than reasoned about:

- The [User Guide](https://docs.pola.rs/user-guide/plugins/expr_plugins/) calls
  expression plugins "the preferred way to create user defined functions... The
  Polars engine will **dynamically link your function at runtime**". The dynamic
  link is the engine `dlopen`-ing our `cdylib`; the plugin itself carries its
  own Polars. The pyo3-polars README says so outright: "The plugin functions are
  **compiled separately**."
- Their canonical `Cargo.toml` is ours: `crate-type = ["cdylib"]`, a plain
  `polars` dependency, `pyo3` with `abi3`, `pyo3-polars` with `derive`.
- Their canonical `lib.rs` opens with
  `#[global_allocator] static ALLOC: PolarsAllocator = PolarsAllocator::new();`
  — **this is prescribed, not an optimisation.** It is the mechanism that keeps
  allocation coherent between the two copies of Polars, and the reason a
  statically linked plugin is safe rather than a double-free waiting to happen.
  Without it we silently ran on a second heap (and 43% slower).

So rule 12 is satisfied for the plugin: it *is* the Rust-native way, because
Polars' extension mechanism is a `dlopen`ed C ABI by design, and nothing on
crates.io publishes a `dylib` to link against anyway (0 of our 453
dependencies; `crate-type` is the publisher's choice).

13. **Know which of the two interfaces a change rides on.** Polars supports two,
    and only one carries a guarantee:
    - **Expression plugin** (`online.ewridge(...)`) — the supported path, with a
      MAJOR/MINOR handshake the loader checks before its first call.
    - **PyO3 extension types** (`PyDataFrame`/`PySeries`, i.e. `ModelBank`) —
      the README states these "are however only provided for convenience and
      **do not have stability guarantees** beyond that the latest definitions
      should work for the latest version of Polars."

    `polars>=1.28.1,<2` in `pyproject.toml` is therefore *measured* for both
    paths but *guaranteed* for neither below the latest — see
    `docs/RELEASE-READINESS.md`. Treat a `ModelBank` break on a new Polars as
    expected maintenance, not a surprise, and check that path first.

## Style

- Rust: `cargo fmt`, `cargo clippy -D warnings`. Small files, one model per file.
- Python: `ruff` (format + lint), type hints, no pandas.
- Docstrings state the math (update equations) for every model.
