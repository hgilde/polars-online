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

## The one rule that matters

**Run `./scripts/gate.sh` before every commit, and let it pass.** It runs
`cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace`,
`ruff format --check`, `ruff check`, a `maturin develop` rebuild, and `pytest`
— in that order, and it prints `gate: PASS` or `gate: FAIL` as its last line.

Two things it exists to prevent, both of which happened before it did:

- Grepping test output for "FAILED" hides compile and lint errors.
- `uv run pytest` does **not** reliably rebuild the extension after a Rust
  change, so the Python suite can silently test a stale binary. The gate
  rebuilds first.

Do not pipe it and then chain a commit — `gate.sh | tail && git commit` commits
even on failure, because a pipeline returns the exit status of `tail`.

## What the tests guarantee

Two properties are load-bearing, and a change that breaks either is wrong by
definition rather than by preference:

1. **Predictions are out-of-sample by construction.** Every row is predicted
   from the state *before* that row's target is folded in.
2. **Chunk invariance.** Feeding a stream as 1 chunk or 1000 chunks produces
   identical output, and so does saving state mid-stream and resuming. The one
   exception is `coef`, a reporting cadence rather than a value.

`crates/online-core/tests/golden.rs` and `tests/test_golden_pipeline.py` pin
exact numbers for both layers. **If a change moves a golden number, that is the
finding** — understand why before regenerating it.

## Conventions

- **Rust**: `cargo fmt`, `cargo clippy -D warnings`, small files, one model per
  file. No `unsafe` in `online-core`. `f64` everywhere.
- **Python**: `ruff` (format + lint), type hints, no pandas.
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
- unit tests with an **oracle** — the recursion written out longhand, an
  equivalent model configured a different way, or the optimality conditions of
  the problem — not just a golden number;
- an entry in `crates/online-core/tests/model_contract.rs`, which checks the
  shared contract (`n_eff` semantics, slot counts, state round-tripping) for
  every model at once;
- wiring in `online-polars/src/spec.rs` and a `po.spec.<name>()` constructor.

## Performance changes

`docs/PERFORMANCE.md` has the measured baseline and the methodology. Measure
before and after with the same scripts, on an idle machine, and put the numbers
in the commit message. Several proposals in there were rejected *because* the
measurement said so; that is a good outcome, not a failed one.

## Commits and PRs

- One logical change per commit, with a message that says what was measured or
  what defect it prevents.
- Reference the task ID where there is one (`P3`, `E12`, `T-W5`).
- Update the relevant doc in the same commit: `docs/PLAN.md`,
  `docs/ENHANCEMENTS.md`, `docs/TESTING.md`, `docs/PERFORMANCE.md`.
