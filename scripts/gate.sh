#!/usr/bin/env bash
# The full check gate, run before every commit. Exits non-zero on any failure
# and prints the actual diagnostics -- grepping this output for "FAILED" hides
# compile and lint errors, which is how two broken commits got through.
#
# Diagnostics are printed *after* the status table, and the last line is always
# a PASS/FAIL banner, so `./scripts/gate.sh | tail -N` still shows the verdict.
# That matters because a pipe discards the exit status unless the caller sets
# `pipefail` -- a third broken commit got through exactly that way. Prefer
# running it unpiped, or chain with `&& git commit`.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

failed=()
diagnostics=""
step() {
    local name="$1"; shift
    printf '%-22s' "$name"
    if out="$("$@" 2>&1)"; then
        echo "OK"
    else
        echo "FAILED"
        failed+=("$name")
        diagnostics+="
--- $name ---
$(echo "$out" | tail -25 | sed 's/^/    /')"
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

if [ ${#failed[@]} -eq 0 ]; then
    echo "gate: PASS"
    exit 0
fi
echo "$diagnostics"
echo "gate: FAIL (${failed[*]})"
exit 1
