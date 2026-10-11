# Contributing

Thanks for looking. This is a small, opinionated library, and the fastest
way to get a change merged is to match the conventions it already has.

## Getting set up

**Install [uv](https://docs.astral.sh/uv/) and a stable Rust toolchain
([rustup](https://rustup.rs)); nothing else is needed.** In each new shell,
run `source scripts/env.sh`, or `. .\scripts\env.ps1` in PowerShell, to
put both on the `PATH`. VS Code's terminal gets them from
`.vscode/settings.json`.

**Run `cargo` through `uv run`, as the commands below do, because
`online-py` builds against pyo3's `abi3-py312`.** That build needs a Python
3.12 or newer interpreter, and `uv run` hands it the project's. The tests
download or generate their own data and cache downloads under `.cache/`.
Offline, the tests that need a download are skipped.

```sh
uv sync                                                # Python env (CPython 3.12 or newer)
source scripts/env.sh                                  # PATH for cargo and uv; `. .\scripts\env.ps1` on Windows
./scripts/gate.sh                                      # every check, the tests' essentials: each commit
./scripts/gate.sh --extended                           # every check and every test: before every push
uv run cargo test --workspace --exclude online-py      # Rust tests; `-- --include-ignored` for all
uv run maturin develop --release -m crates/online-py/Cargo.toml   # the extension, into the env
uv run pytest                                          # Python tests
uv run --group docs sphinx-build -W docs/reference docs/_build/html   # API reference
uv run python scripts/validate.py > docs/VALIDATION.md # re-run the experiments that chose the defaults
uv run python scripts/regime_experiments.py all        # the docs/REGIMES.md experiments
uv run python scripts/benchmark.py                     # throughput
uv run python scripts/windows_bench.py                 # the window operators against Polars' rolling recipe
uv run python scripts/sklearn_comparison.py            # the comparison with scikit-learn
uv run python scripts/compare_release.py               # every output against the newest release's, bit for bit
uv run pytest -m soak                                  # the opt-in 10M-row soak
./scripts/mutants.sh --in-diff <(git diff main...)     # mutation testing of the code a branch touches
```

## The gate

**Run `./scripts/gate.sh` before every commit, and let it pass. Run
`./scripts/gate.sh --extended` before every push, always.** The gate has two
tiers of tests ([docs/TESTING.md, "Two tiers"](docs/TESTING.md#two-tiers)).
By default it runs the essentials, which keep the many commits of a task in
progress quick. With `--extended` it runs every test, as CI and a release
do, and what reaches GitHub has passed everything locally. Both run these
steps, in this order:

1. `cargo fmt --check`
2. `cargo clippy -D warnings`
3. `cargo test --workspace --exclude online-py`, since online-py has no Rust
   tests and would link libpython into every test binary. `--extended` adds
   `-- --include-ignored`, which runs the tests marked
   `#[ignore = "extended: ..."]`
4. `uv lock --check`, before any `uv run` could rewrite a stale lock file
5. `ruff format --check`
6. `ruff check`
7. `mypy`
8. a `maturin develop` rebuild. `uv run pytest` does **not** reliably
   rebuild the extension after a Rust change, so without this step the
   Python suite can silently test a stale binary.
9. `pytest`, which also runs every README code block, every docstring
   example and the structure check of every Markdown file. The essentials
   leave out the tests marked `extended`, and draw fewer examples in the
   property tests
10. a `sphinx-build -W` of the API reference, so a docstring that is not
    valid RST fails the gate, not the docs deploy.

Its first line names the tier, and its last line is `gate: PASS` or
`gate: FAIL` with the tier beside it. The essentials' verdict says it is not
the pre-push check. The gate exists because grepping test output for
"FAILED" hides compile and lint errors.

**Do not pipe it and then chain a commit.** `gate.sh | tail && git commit`
commits even on failure, because a pipeline returns the exit status of
`tail`.

## What a change must not break

**A change that breaks either of the library's two guarantees is wrong by
definition.** Every row is predicted before its own target is learned, and
a stream gives the same numbers in one chunk or a thousand, with or
without a save and resume in the middle. The README's [Row order and the
two guarantees](README.md#row-order-and-the-two-guarantees) states both,
and says which rows carry `coef` and `support_coef`, the one output that
moves with the chunking.

**If a change moves a golden number, that is the finding, so understand
why before regenerating it.** `crates/online-core/tests/golden.rs` and
`tests/test_golden_pipeline.py` pin the numbers both layers produce, to
within 1e-12 on every platform.

## Conventions

- **Rust**: `cargo fmt`, `cargo clippy -D warnings`, small files, one model per
  file. No `unsafe` in `online-core`. `f64` everywhere.
- **Python**: `ruff` (format + lint), `mypy` clean, type hints. The package
  depends on `polars` alone. A test may use any open-source library the
  package does not depend on, declared in the dev group and imported
  plainly; `tests/test_dependency_policy.py` checks it.
- **Docstrings state the math.** Every model's docs carry its update equations.
- **Documentation follows [`docs/WRITING.md`](docs/WRITING.md)**, the README,
  the guides and the docstrings alike. Some docstring text is pinned by tests,
  line wraps included, so run the doc tests after rewording one.
- **No data files in the repo, ever.** Tests generate or download what they
  need, cached under the gitignored `.cache/`. `tests/test_repo_hygiene.py`
  enforces this.
- **Comments explain why a line is there.** The house style is a comment
  that names the bug a guard prevents.

## Adding a model

Models live in `crates/online-core/src/`, behind the `OnlineModel` trait, and
know nothing about Polars, Python, or clocks-as-columns. Plumbing lives in
`online-polars` and `online-py`. A new model needs:

- the recursion, with the update equations in the docstring;
- unit tests with an **oracle**, not just a golden number. That is another
  library that computes the same quantity wherever one does; otherwise the
  recursion written out longhand, an equivalent model configured a different
  way, or the optimality conditions of the problem;
- an entry in `crates/online-core/tests/model_contract.rs`, which checks the
  shared contract (`weight_sum` semantics, slot counts, state round-tripping) for
  every model at once;
- wiring in `online-polars/src/spec.rs` and a `po.spec.<name>()` constructor.

That is the shape of it. [`docs/EXTENDING.md`](docs/EXTENDING.md) is the full
list of places a model touches, in order, with the test that fails when one is
skipped. `tests/test_model_registry.py` holds the per-model test lists to
the Rust registry (`ModelKind::KINDS`), so a half-wired model fails the gate
rather than going unswept. It holds the release comparison's workload and
`_spec.UNSUPERVISED` as well.

## Performance changes

**Measure before and after, with the same scripts, on an idle machine, and
put the numbers in the commit message.** `docs/PERFORMANCE.md` has the
measured baseline and the methodology. Before a release, compare the head
against the last release's wheel: `scripts/compare_release.py` for the
numbers, bit for bit, and the benchmark for the speed.

**A proposal the measurement rejects is a good outcome.** Several in
`docs/PERFORMANCE.md` were rejected *because* the measurement said so, and
its §5 collects them.

## Commits and pull requests

- One logical change per commit, with a message that says what was measured or
  what defect it prevents.
- Open the message with the piece of work the commit belongs to, as the
  history does: its number, or the review and the finding IDs it answers.
- Update the relevant doc in the same commit: the README,
  `CHANGELOG.md`'s unreleased entry, `docs/TESTING.md`,
  `docs/PERFORMANCE.md`, `docs/OUTPUTS.md` regenerated where an output
  changed, and the design record a new feature's design belongs in. A
  change to the public API shows as a diff of `tests/api_surface.txt`,
  which the pull request includes. [`docs/README.md`](docs/README.md) says
  which document holds what, the records included.
- Keep section numbers where they are — the code and the README cite them.

## Licensing of contributions

This project is Apache-2.0. Contributions follow the standard inbound =
outbound convention: by submitting a pull request you agree that your
contribution is licensed under Apache-2.0, per §5 of the license. There is no
CLA to sign and no DCO bot; the pull request itself is the record.
