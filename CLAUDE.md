# polars-online

Streaming / online regression models for Polars. Rust core, exposed as
(1) a chunk-fed Python "model bank", which also takes a `LazyFrame` and chunks
it, and runs as a streaming `LazyFrame` plan; and (2) a standalone Rust CLI for
deployment. Runs on data that does not fit in memory. Every surface streams:
the expression plugin was removed in task 85 because polars hands a stateful
user expression its whole column, which is the opposite of the point.

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
crates/online-polars/  Rust-side integration: the model bank over Arrow chunks (a polars adapter
                       builds one from a frame), and the runner over parquet streams
crates/online-cli/     binary: parquet in -> parquet out, config from TOML
crates/online-py/      pyo3 + pyo3-polars: the Python ModelBank class
python/polars_online/  Python package (thin wrappers, frame namespaces)
tests/                 pytest (Python) — integration, invariance, oracle tests
docs/PLAN.md           design + task list
docs/reference/        Sphinx API reference over the docstrings (RST dialect); -W in gate and CI
docs/EXTENDING.md      every place a new model touches, with the test that catches each omission
```

## Commands

```
uv sync                                  # Python env
cargo test --workspace --exclude online-py   # Rust tests (online-py has none; see ci.yml)
maturin develop --release -m crates/online-py/Cargo.toml
uv run pytest -x                         # Python tests (downloads/generates data on first run)
uv run --group docs sphinx-build -W docs/reference docs/_build/html   # API reference (gate + CI)
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
   now checks every model against the same recursion. **Under a `window`, `n_eff` is the
   weight inside the window** -- in the field a model emits, in its `min_periods` gate,
   and in its accessor -- because that is what the fit is read from (review 2026-09-12,
   pattern D; `lasso`, `ew_class`, `marginal` and `ew_cov`'s accessor each said otherwise
   in one of those three places). A model whose coefficients do not decay (`pa`, `sgd`)
   still decays `n_eff`, so after a gap `min_periods` can withhold a fit exactly as good as
   before it; `ewridge`'s mean-form fit does not move on a gap either, so this is the
   library's convention, not one model's quirk (review 2026-09-12, D10). Each target's
   `min_periods` is checked against that target's own weight where the model keeps one
   (S2); the emitted `n_eff` is the shared weight either way.
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

So rule 12 is satisfied, and **stays** satisfied now that the plugin is gone
(task 85). The quotations above are the evidence that settled the question,
not a description of a surface we still ship; what they established still
holds of the wheel itself. `online-py` is a `cdylib` that Python `dlopen`s, it
carries its own statically linked Polars, and the `PolarsAllocator` above is
still what keeps allocation coherent between the two copies — which is why
that `#[global_allocator]` line must stay whatever else changes. Nothing on
crates.io publishes a `dylib` to link against anyway (0 of our 453
dependencies; `crate-type` is the publisher's choice).

13. **Know which of the interfaces a change rides on.** Three, and the two
    that are py-polars' to change carry no guarantee. (The expression plugin —
    which had a version handshake — was removed in task 85: it read every row
    by construction, which is the opposite of what this library is for.)
    - **PyO3 extension types** (`PyDataFrame`/`PySeries`, i.e. `ModelBank`) —
      the README states these "are however only provided for convenience and
      **do not have stability guarantees** beyond that the latest definitions
      should work for the latest version of Polars."

    - **IO plugin** (`lf.online.fit_predict(...)`, `polars.io.plugins.register_io_source`)
      — documented in the user guide but decorated `@unstable` in py-polars:
      "may be changed at any point without it being considered a breaking
      change". The bank is the source; polars does not re-apply the
      projection, predicate or slice it pushes into a Python source, so the
      source honours all three (`python/polars_online/_frame.py`).

    - **The Arrow PyCapsule interface** (`ModelBank.fit_predict_arrow`, task
      86) — the *output* side only, and the one path whose contract is not
      py-polars' to change: `__arrow_c_array__` is an Arrow specification,
      py-polars consumes it through the public `PySeries.from_arrow_c_array`,
      and pyarrow and duckdb consume it too. A break here would be an
      Arrow-level break rather than a polars one. Two caveats, both real: the
      *input* still arrives as a `PyDataFrame`, so a `fit_predict_arrow` call
      rides on this **and** on the first entry above; and which py-polars
      versions expose `from_arrow_c_array` has not been measured here, so the
      floor below is not known to hold for it.

    `polars>=1.34.0,<3` in `pyproject.toml` is therefore *measured* for all
    three but *guaranteed* for none below the latest — and the two that stream
    are among those — see `docs/RELEASE-READINESS.md`.
    Treat a `ModelBank` or IO-plugin break on a new Polars as expected
    maintenance, not a surprise, and check those paths first. The floor is
    `LazyFrame.collect_batches` (py-polars 1.34.0), which the IO plugin reads
    with; `ModelBank` alone works from 1.28.1.

## Style

- Rust: `cargo fmt`, `cargo clippy -D warnings`. Small files, one model per file.
- Python: `ruff` (format + lint), `mypy` clean (`uv run mypy`), type hints, no pandas
  in the package (tests may use it as an oracle).
- Docstrings state the math (update equations) for every model.
