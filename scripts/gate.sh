#!/usr/bin/env bash
# The check gate, in two tiers (docs/TESTING.md, "Two tiers"):
#
#   ./scripts/gate.sh              the essentials: every commit while a task
#                                  is in progress. Every check, and the
#                                  tests' essentials tier.
#   ./scripts/gate.sh --extended   everything: before every push, always,
#                                  whatever the essentials said. CI and a
#                                  release run the same tier.
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
case "${1:-}" in
    "") ;;
    --extended) tier=extended ;;
    *) echo "usage: scripts/gate.sh [--extended]" >&2; exit 2 ;;
esac

# What each tier runs. Every proptest property names its count as a multiple
# of proptest's own, 256 unless PROPTEST_CASES says otherwise, so the
# essentials' 32 runs each at an eighth: the generated contract streams at
# 16 rather than 128, the clock's at 256 rather than 2,048.
if [ "$tier" = extended ]; then
    export HYPOTHESIS_PROFILE=extended
    unset PROPTEST_CASES
    cargo_tier=(-- --include-ignored)
    pytest_tier=()
else
    export HYPOTHESIS_PROFILE=essential
    export PROPTEST_CASES=32
    cargo_tier=()
    pytest_tier=(-m "not extended and not soak")
fi
echo "gate: the $tier tier"

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

step "cargo fmt"   cargo fmt --all -- --check
step "cargo clippy" cargo clippy --workspace --all-targets -- -D warnings
# online-py is left out: it has no Rust tests, and with it in the build every
# test binary links libpython (.github/workflows/ci.yml says why that fails).
# The time printed includes the test build when Rust changed. The `+`
# expansions keep an empty tier's arguments empty under macOS's bash 3.2,
# where `set -u` calls an empty array unbound.
step "cargo test"   cargo test --workspace --exclude online-py ${cargo_tier[@]+"${cargo_tier[@]}"}
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
step "pytest"       uv run pytest -q ${pytest_tier[@]+"${pytest_tier[@]}"}
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
