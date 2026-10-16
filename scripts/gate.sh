#!/usr/bin/env bash
# The check gate, in two tiers (docs/TESTING.md, "Two tiers"):
#
#   ./scripts/gate.sh              the essentials: every commit while a task
#                                  is in progress. Every check, and the
#                                  tests' essentials tier.
#   ./scripts/gate.sh --extended   everything: before every push, always,
#                                  whatever the essentials said. CI and a
#                                  release run the same tier.
#   ./scripts/gate.sh --serial     either tier, every test in one process,
#                                  as the gate ran before it ran in parallel
#                                  (`--extended --serial` for both).
#
# The tests run in parallel (docs/TESTING.md, "Iterating"): cargo-nextest
# runs each Rust test as a process of its own, where it is installed, and
# pytest-xdist runs the Python tests in several processes. Both start as
# many as `scripts/workers.py` says the machine has cores free (2 to 8;
# GATE_WORKERS overrides it). The Python tests marked `serial`, which read
# their process's memory, run after the rest in one process, alone. nextest
# does not run doctests, so `cargo test --doc` runs them beside it. The same
# tests run either way, with the same results.
#
# The essentials leave out the tests marked `extended` (pytest
# `-m "not extended and not soak"`, cargo test without
# `--include-ignored`) and run the property tests at fewer examples
# (HYPOTHESIS_PROFILE=essential, PROPTEST_CASES below). The two test steps
# print their wall time, so a slow test creeping into the essentials shows.
#
# Exits non-zero on any failure and prints the actual diagnostics -- grepping
# this output for "FAILED" hides compile and lint errors, which is how two
# broken commits got through.
#
# Diagnostics are printed *after* the status table, and the last line is always
# a PASS/FAIL banner naming the tier, so `./scripts/gate.sh | tail -N` still
# shows the verdict. That matters because a pipe discards the exit status
# unless the caller sets `pipefail` -- a third broken commit got through
# exactly that way. Prefer running it unpiped, or chain with `&& git commit`.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

tier=essentials
serial=
for arg in "$@"; do
    case "$arg" in
        --extended) tier=extended ;;
        --serial) serial=1 ;;
        *) echo "usage: scripts/gate.sh [--extended] [--serial]" >&2; exit 2 ;;
    esac
done
workers=$(python3 scripts/workers.py)
if [ "$workers" -le 1 ]; then serial=1; fi
nextest=
if [ -z "$serial" ] && cargo nextest --version >/dev/null 2>&1; then nextest=1; fi

# What each tier runs. Every proptest property names its count as a multiple
# of proptest's own, 256 unless PROPTEST_CASES says otherwise, so the
# essentials' 32 runs each at an eighth: the generated contract streams at
# 16 rather than 128, the clock's at 256 rather than 2,048.
if [ "$tier" = extended ]; then
    export HYPOTHESIS_PROFILE=extended
    unset PROPTEST_CASES
    cargo_tier=(-- --include-ignored)
    nextest_tier=(--run-ignored all)
    pytest_tier=()
    marks="not soak"
else
    export HYPOTHESIS_PROFILE=essential
    export PROPTEST_CASES=32
    cargo_tier=()
    nextest_tier=()
    pytest_tier=(-m "not extended and not soak")
    marks="not extended and not soak"
fi
if [ -n "$serial" ]; then
    echo "gate: the $tier tier, in one process"
else
    echo "gate: the $tier tier, in $workers processes${nextest:+ (cargo-nextest)}"
fi

failed=()
diagnostics=""
step() {
    local name="$1"; shift
    printf '%-22s' "$name"
    local start=$SECONDS
    if out="$("$@" 2>&1)"; then
        echo "OK ($((SECONDS - start)) s)"
    else
        echo "FAILED ($((SECONDS - start)) s)"
        failed+=("$name")
        diagnostics+="
--- $name ---
$(echo "$out" | tail -25 | sed 's/^/    /')"
    fi
}

pytest_or_none() {
    uv run pytest "$@"
    local rc=$?
    # Exit 5 is pytest's "no test selected". The essentials hold no `serial`
    # test today, so there that is not a failure; the extended tier holds the
    # memory tests, so there an empty selection is a mark gone wrong.
    [ "$rc" -eq 5 ] && [ "$tier" = essentials ] && return 0
    return "$rc"
}

step "cargo fmt"   cargo fmt --all -- --check
step "cargo clippy" cargo clippy --workspace --all-targets --locked -- -D warnings
# online-py is left out: it has no Rust tests, and with it in the build every
# test binary links libpython (.github/workflows/ci.yml says why that fails).
# `--locked` on every cargo step, as CI's: without it cargo re-resolves a
# Cargo.lock that no longer matches the manifests, silently, and the
# rewritten lock rides into the commit.
# The time printed includes the test build when Rust changed. The `+`
# expansions keep an empty tier's arguments empty under macOS's bash 3.2,
# where `set -u` calls an empty array unbound.
if [ -n "$nextest" ]; then
    step "cargo nextest"  cargo nextest run --workspace --exclude online-py --locked --no-fail-fast \
        --test-threads "$workers" ${nextest_tier[@]+"${nextest_tier[@]}"}
    step "cargo doctests" cargo test --workspace --exclude online-py --doc --locked
else
    step "cargo test"   cargo test --workspace --exclude online-py --locked ${cargo_tier[@]+"${cargo_tier[@]}"}
fi
# Before the first `uv run`, which re-locks by default and would rewrite a
# stale uv.lock before the check could see it.
step "uv lock"      uv lock --check
step "ruff format"  uv run ruff format --check .
step "ruff check"   uv run ruff check .
step "mypy"         uv run mypy
# The extension MUST be rebuilt before pytest. `uv run pytest` re-syncs the
# project but does not reliably pick up a Rust change, so without this the
# Python suite can silently test a stale binary -- which once made a working
# feature look like a no-op, because both branches of the comparison ran the
# same old code.
step "maturin develop" uv run maturin develop --release -m crates/online-py/Cargo.toml
if [ -n "$serial" ]; then
    step "pytest"       uv run pytest -q ${pytest_tier[@]+"${pytest_tier[@]}"}
else
    step "pytest"       uv run pytest -q -n "$workers" --dist worksteal -m "$marks and not serial"
    step "pytest serial" pytest_or_none -q -m "$marks and serial"
fi
# The API reference imports the package, so it runs after the build. -W: a
# docstring that is not valid reStructuredText fails here, not on the site.
step "sphinx"       uv run --group docs sphinx-build -W -q docs/reference docs/_build/html

if [ "$tier" = extended ]; then
    banner="the extended tier: the full suite, the pre-push check"
else
    banner="the essentials, not the pre-push check: run scripts/gate.sh --extended before every push"
fi
echo "gate: $SECONDS s in all"
if [ ${#failed[@]} -eq 0 ]; then
    echo "gate: PASS ($banner)"
    exit 0
fi
echo "$diagnostics"
echo "gate: FAIL (${failed[*]}) ($banner)"
exit 1
