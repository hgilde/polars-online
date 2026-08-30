#!/usr/bin/env bash
# Report test coverage. Reported, never gating (docs/TESTING.md T-D4).
#
#   ./scripts/coverage.sh
#
# Note the two numbers measure different things and neither is the whole story:
# `cargo llvm-cov` sees only what `cargo test` executes, so the crates driven
# mainly from Python (online-py, and much of online-polars) look far emptier
# than they are -- the pytest suite exercises them through the compiled
# extension, which the Rust instrumentation cannot observe.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "=== Rust (cargo test only; see the note above) ==="
cargo llvm-cov --workspace --summary-only

echo
echo "=== Python (drives the extension end to end) ==="
uv run pytest --cov=polars_online --cov-report=term-missing -q
