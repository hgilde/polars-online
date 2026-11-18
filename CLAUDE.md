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
scripts/gate.sh                          # the essentials: each commit while a task is in progress
scripts/gate.sh --extended               # everything: before EVERY push, always
cargo test --workspace --exclude online-py   # Rust essentials only (online-py has none; see ci.yml)
cargo test --workspace --exclude online-py -- --include-ignored   # every Rust test
maturin develop --release -m crates/online-py/Cargo.toml
uv run pytest -x                         # every Python test (downloads/generates data on first run)
HYPOTHESIS_PROFILE=essential uv run pytest -m "not extended and not soak"   # Python essentials
uv run --group docs sphinx-build -W docs/reference docs/_build/html   # API reference (gate + CI)
cargo run -p online-cli -- --config examples/bank.toml
scripts/quick.sh                         # inside a task: tests for what the branch changed (dev-wheel build)
scripts/gate.sh --serial                 # either tier in one process (the gate runs tests in parallel)
scripts/clear_caches.sh --worktrees      # build caches: after each push wave, or under 100 GB free
scripts/slot.sh take <branch> [<base>]   # a warm worktree slot for a task (give / list / overlap)
scripts/budgets.py check|raises|apply    # performance and memory budgets (budgets.toml)
scripts/oracles.py check|raises|apply    # the oracle ledger (oracles.toml): each grid's error only tightens
uv run python scripts/oracles.py rerun   # the oracle grids whose hashes the gate's guard says moved
```

Workflow (the detail and each rule's reason: `docs/records/AGENT-WORKFLOW.md`):
- Workers iterate with `scripts/quick.sh` and finish with `scripts/quick.sh --extended`
  plus the lints, never a full gate, never a merge of `main` unless asked.
- The coordinator merges a wave of 2+ branches onto `integration`; a wave of one merges onto
  `main` directly. **The full tier runs before every push, on the exact sha**, and is the
  only gate a wave pushed at once gets; the essentials gate runs on `integration` only when
  the wave will wait before its push (the user, 2026-11-14). Budget tightenings from CI ride
  with the next wave, never a gate of their own. A gated wave with no open bug is pushed
  without asking. A release dispatch needs the user's word; PyPI approval is the user's click.
- At most two building agents (three while the 1-minute load is under about 20), each with
  `CARGO_BUILD_JOBS=6`, pytest `-n 4`, `GATE_WORKERS=4`; check `uptime` first.
- No local full `cargo mutants`: CI's changed-lines pass is the one check; a survivor fix
  is checked locally only by `--in-diff`, alone, under `.claude/coord/mutants.lock`.
- Coordination files live in `.claude/coord/`, never `/tmp`. Workers get a slot
  (`scripts/slot.sh take <branch> main`, `give <n>` after the merge), not a fresh worktree.
- After each push wave, `scripts/clear_caches.sh --worktrees` (or under 100 GB free). sccache off.

Two tiers of tests (docs/TESTING.md, "Two tiers"; PLAN task 210). The
essentials gate each commit while a task is in progress, for speed of
iteration; **the full suite runs before every push, always**
(`scripts/gate.sh --extended`), whatever the essentials said, and CI and a
release run it too. The defaults are asymmetric: a plain `uv run pytest`
runs both tiers, but a plain `cargo test` skips the extended Rust tests
(`#[ignore = "extended: ..."]`), so the full Rust run is `cargo test --
--include-ignored`.

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
   on any layout change. Every schema from `MIN_SCHEMA_VERSION` / `MIN_BANK_SCHEMA_VERSION`
   to `SCHEMA_VERSION` has frozen fixtures (`crates/*/tests/state_fixtures.rs`): the
   current schema's load, continue to the bit and re-save byte for byte; a previous
   schema's load through its loader, continue to the bit and re-save as the current
   schema's fixture bytes of the same case (`state_fixtures/v<N>/`, found by each crate's `build.rs`).
   A layout change cannot land without them, and a fixture must freeze every form a
   layout can take (a test decodes each file and holds the new forms non-empty; review
   5, B1 and D3). Pre-1.0 (the user, 2026-09-14 and 2026-09-28): a state saved before
   the next release need not load, so regenerate the fixtures (`PRINT_STATE_FIXTURES=1`)
   and raise both minimums instead of writing a loader. From 1.0 (decided 2026-10-06,
   PLAN §18 D1): the minimums stay at 1.0's schema, and each layout change copies the
   current sets to `v<N>/`, ships a loader and regenerates the current sets. The bank
   fixture carries the package version, so it is regenerated at every version bump.
   `ModelState`'s variant names are frozen: they are the file's tags.
   Fifteen exceptions, each the user's, are numbered and stated in full in
   `docs/records/STATE-CONTINUATION.md` ("rule 5's Nth exception" means its Nth);
   a new one is added there and counted here.
6. No `unsafe` in `online-core`. f64 everywhere.
7. Commit after each completed task in `docs/PLAN.md`, with the task number in the message.
8. **`n_eff` -- the models' accessor, emitted as `weight_sum` -- means the same thing in
   every model**: the accumulated weight *before* this
   row's update and *before* its own decay. That is what makes `min_weight` portable
   across a bank. `sgd` and `pa` once applied the row's decay first, so `min_weight`
   quietly meant a different number of rows for them; `crates/online-core/tests/model_contract.rs`
   now checks every model against the same recursion. **Under a `window_size`, `n_eff` is the
   weight inside the window** -- in the field a model emits, in its `min_weight` gate,
   and in its accessor -- because that is what the fit is read from (review 2026-09-12,
   pattern D; `lasso`, `ew_class`, `marginal` and `ew_cov`'s accessor each said otherwise
   in one of those three places). A model whose coefficients do not decay (`pa`, `sgd`)
   still decays `n_eff`, so after a gap `min_weight` can withhold a fit exactly as good as
   before it; `ewridge`'s mean-form fit does not move on a gap either, so this is the
   library's convention, not one model's quirk (review 2026-09-12, D10). Each target's
   `min_weight` is checked against that target's own weight where the model keeps one
   (S2); the emitted `weight_sum` is the shared weight either way.
9. **A zero-weight row is legal** and means "advance the clock, learn nothing" — including
   as the *first* row of a stream, where `lam*w_sum + w` is 0 and the mean-form update's
   `a` and `b` are both 0/0. Guard every such division; an unguarded one poisons the state
   with a NaN that never washes out.

- **Budgets only tighten** (`budgets.toml`; task 251, the user, 2026-10-11:
  "we want to avoid slipping by 5% repeatedly"). A failure says *over its
  budget*: find the cause; if it buys something worth having, raise the
  budget in the same change, with a `[[raised]]` entry giving the reason and
  quoting a CHANGELOG *Performance* entry (`scripts/budgets.py raises` holds
  this over the file's history). Or it says *tighten the budget to N*: run
  `python scripts/budgets.py apply` on the run's `measured.json` (the
  Budgets artifact) or `memory.json` (`POLARS_ONLINE_BUDGETS_OUT`), so the
  gain is kept. Instruction counts come from CI's Linux Valgrind;
  `cargo test -p online-bench` runs the cases natively. A proposal that
  changes a data layout, a per-row path or a default that adds work carries
  its measured memory and speed cost (the coef-as-Array lesson, 2026-10-11).

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
    The rule covers production deployments -- the wheel, the CLI binary --
    not tests: a dev-dependency linked only into test binaries needs no
    raise (the user, 2026-09-25).

Static linking of Polars into the wheel is what Polars prescribes for a compiled
extension, and the `#[global_allocator] static ALLOC: PolarsAllocator` line in
`online-py` must stay: it keeps allocation coherent between the two copies of Polars
(the evidence, raised and resolved 2026-08-31: `docs/records/LINKING.md`).

13. **Know which of the interfaces a change rides on**: the PyO3 extension types
    (`ModelBank`), the IO plugin (`lf.online.fit_predict`, `@unstable` in py-polars),
    an expression's serialized form (the window formulas, `_formula.py`), and the Arrow
    PyCapsule output. The first three carry no stability guarantee from py-polars:
    treat a break on a new Polars as expected maintenance and check those paths first.
    `polars>=1.37.0,<3` is measured, not guaranteed; 1.37.0 is the first whose
    `collect_batches` does not read ahead of a slower consumer. Each interface's
    account and the read-ahead measurements: `docs/records/LINKING.md`.

## Style

- Rust: `cargo fmt`, `cargo clippy -D warnings`. Small files, one model per file.
- Python: `ruff` (format + lint), `mypy` clean (`uv run mypy`), type hints, no pandas
  in the package.
- **Tests may use libraries the package does not depend on**, when a test needs
  one: an oracle, a second opinion, or interop with that library (the user,
  2026-09-24). Declare it in the dev group by name and import it plainly, with
  no `importorskip`. Never add it to the package's dependencies, which stay
  `polars`. `docs/TESTING.md`, "Libraries the package does not depend on", has
  the rules, and `tests/test_dependency_policy.py` checks them.
- **An oracle comes from a third-party library wherever one computes the
  quantity** (the user, 2026-09-27: "Is there no third party oracle
  library?"). Write only the definition by hand, from the paper; let the
  library compute it. In Rust tests, the only ones `cargo mutants` sees, use
  `crate::oracle` (`crates/online-core/src/oracle.rs`, compiled for tests
  only; the user, 2026-10-06): faer's partial-pivot LU and self-adjoint
  eigensolver, which share no arithmetic with `solve.rs`'s Cholesky. Add a
  helper there rather than computing a matrix result in a test module. Its
  eigensolver is the one `Pca::of` and `rcov`'s PSD repair run, so it is no
  oracle for them. In Python, scikit-learn, scipy or statsmodels, in
  `tests/test_second_opinion.py`. Where the library is Python's alone, write
  both. `docs/TESTING.md` has the rule beside the one above.
- Docstrings state the math (update equations) for every model.
- **No reader-facing doc references the plan** (the user, 2026-10-10: "No doc
  should reference the plan, everything should be explained from first
  principals"). The README, every guide in `docs/README.md`'s Guides table,
  CONTRIBUTING, `llms.txt`, every public docstring and every error message
  cite no `docs/PLAN.md` section, task number, review or record (REVIEW-*,
  ENHANCEMENTS, the records under `docs/records/`, and the rest that
  `docs/README.md` files as records). State the reason itself: the mechanism,
  the measurement and how it was made, the trade-off. A link to another guide
  that explains the point is fine; a link to a record is not. Code comments,
  tests, the CHANGELOG and the records themselves may keep task numbers.
  `tests/test_no_plan_references.py` holds it, and its file list grows as
  each doc is rewritten.
