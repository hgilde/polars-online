# Contributing

Thanks for looking. This is a small, opinionated library; the fastest way to
get a change merged is to match the conventions it already has.

## Getting set up

```sh
uv sync                                    # Python env (CPython 3.12+)
source scripts/env.sh                      # PATH for cargo/uv; `. .\scripts\env.ps1` on Windows
./scripts/gate.sh                          # everything CI checks, in one command
```

Prerequisites are [uv](https://docs.astral.sh/uv/) and a stable Rust toolchain
([rustup](https://rustup.rs)). Nothing else.

## The gate

**Run `./scripts/gate.sh` before every commit, and let it pass.** It runs
these steps, in this order:

1. `cargo fmt --check`
2. `cargo clippy -D warnings`
3. `cargo test --workspace --exclude online-py`, since online-py has no Rust
   tests and would link libpython into every test binary
4. `ruff format --check`
5. `ruff check`
6. `mypy`
7. a `maturin develop` rebuild. `uv run pytest` does **not** reliably
   rebuild the extension after a Rust change, so without this step the
   Python suite can silently test a stale binary.
8. `pytest`
9. a `sphinx-build -W` of the API reference, so a docstring that is not
   valid RST fails the gate, not the docs deploy.

Its last line is `gate: PASS` or `gate: FAIL`. It exists because grepping
test output for "FAILED" hides compile and lint errors.

**Do not pipe it and then chain a commit.** `gate.sh | tail && git commit`
commits even on failure, because a pipeline returns the exit status of
`tail`.

## What the tests guarantee

Two properties are load-bearing, and a change that breaks either is wrong by
definition rather than by preference:

1. **Predictions are out-of-sample by construction.** Every row is predicted
   from the state *before* that row's target is folded in.
2. **Chunk invariance.** Feeding a stream as 1 chunk or 1000 chunks produces
   identical output, and so does saving state mid-stream and resuming. The one
   exception is `coef`, a reporting cadence rather than a value.

`crates/online-core/tests/golden.rs` and `tests/test_golden_pipeline.py` pin
the numbers both layers produce, to within 1e-12 on every platform. **If a
change moves a golden number, that is the finding.** Understand why before
regenerating it.

## Conventions

- **Rust**: `cargo fmt`, `cargo clippy -D warnings`, small files, one model per
  file. No `unsafe` in `online-core`. `f64` everywhere.
- **Python**: `ruff` (format + lint), `mypy` clean, type hints. No pandas in
  the package; tests may use it as an oracle.
- **Docstrings state the math.** Every model's docs carry its update equations.
- **No data files in the repo, ever.** Tests generate or download what they
  need, cached under the gitignored `.cache/`. `tests/test_repo_hygiene.py`
  enforces this.
- **Comments explain why, not what.** The codebase is full of comments naming
  the bug a guard prevents; that is the house style.

## Adding a model

Models live in `crates/online-core/src/`, behind the `OnlineModel` trait, and
know nothing about Polars, Python, or clocks-as-columns. Plumbing lives in
`online-polars` and `online-py`. A new model needs:

- the recursion, with the update equations in the docstring;
- unit tests with an **oracle**, not just a golden number: the recursion
  written out longhand, an equivalent model configured a different way, or
  the optimality conditions of the problem;
- an entry in `crates/online-core/tests/model_contract.rs`, which checks the
  shared contract (`n_eff` semantics, slot counts, state round-tripping) for
  every model at once;
- wiring in `online-polars/src/spec.rs` and a `po.spec.<name>()` constructor.

That is the shape of it. [`docs/EXTENDING.md`](docs/EXTENDING.md) is the full
list of places a model touches, in order, with the test that fails when one is
skipped. `tests/test_model_registry.py` holds every per-model list to the Rust
registry (`ModelKind::KINDS`), so a half-wired model fails the gate rather than
going unswept.

## Performance changes

**Measure before and after, with the same scripts, on an idle machine, and
put the numbers in the commit message.** `docs/PERFORMANCE.md` has the
measured baseline and the methodology.

**A proposal the measurement rejects is a good outcome.** Several in
`docs/PERFORMANCE.md` were rejected *because* the measurement said so, and
its §5 collects them.

## Commits and pull requests

- One logical change per commit, with a message that says what was measured or
  what defect it prevents.
- Reference the task ID where there is one (`P3`, `E12`, `T-W5`).
- Update the relevant doc in the same commit: `docs/PLAN.md`,
  `docs/ENHANCEMENTS.md`, `docs/TESTING.md`, `docs/PERFORMANCE.md`.
  [`docs/README.md`](docs/README.md) says which document holds what.
- Keep section numbers where they are — the code and the README cite them.

## Licensing of contributions

This project is Apache-2.0. Contributions follow the standard inbound =
outbound convention: by submitting a pull request you agree that your
contribution is licensed under Apache-2.0, per §5 of the license. There is no
CLA to sign and no DCO bot; the pull request itself is the record.
