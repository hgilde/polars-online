#!/usr/bin/env bash
# The full check gate, run before every commit. Exits non-zero on the first
# failure and prints the actual diagnostics -- grepping this output for "FAILED"
# hides compile and lint errors, which is how two broken commits got through.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

fail=0
step() {
    local name="$1"; shift
    printf '%-22s' "$name"
    if out="$("$@" 2>&1)"; then
        echo "OK"
    else
        echo "FAILED"
        echo "$out" | tail -25 | sed 's/^/    /'
        fail=1
    fi
}

step "cargo fmt"   cargo fmt --all -- --check
step "cargo clippy" cargo clippy --workspace --all-targets -- -D warnings
step "cargo test"   cargo test --workspace
step "ruff format"  uv run ruff format --check .
step "ruff check"   uv run ruff check .
# The extension MUST be rebuilt before pytest. `uv run pytest` re-syncs the
# project but does not reliably pick up a Rust change, so without this the
# Python suite can silently test a stale binary -- which once made a working
# feature look like a no-op, because both branches of the comparison ran the
# same old code.
step "maturin develop" uv run maturin develop --release -m crates/online-py/Cargo.toml
step "pytest"       uv run pytest -q

exit "$fail"
