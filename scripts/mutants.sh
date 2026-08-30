#!/usr/bin/env bash
# Mutation testing on online-core (docs/TESTING.md T-D4).
#
#   ./scripts/mutants.sh                      # the whole crate (slow: ~1645 mutants)
#   ./scripts/mutants.sh crates/online-core/src/clock.rs   # one file (~1 min)
#
# What it does: makes one small change to the source (flip an operator, replace
# a function body with a constant), rebuilds, and reruns the tests. A mutant
# that is "caught" means some test failed -- good. A mutant that is "MISSED"
# means every test still passed with the code deliberately broken, which is a
# gap in the tests, not a bug in the code.
#
# Scoped to online-core on purpose: cargo-mutants only runs `cargo test`, so it
# cannot see the pytest suite that exercises online-polars and online-py through
# the compiled extension. Mutants there would report as missed regardless.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

args=(--package online-core -j 4)
for f in "$@"; do args+=(--file "$f"); done
cargo mutants "${args[@]}"
