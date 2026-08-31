#!/usr/bin/env bash
# Mutation testing on online-core (docs/TESTING.md T-D4).
#
#   ./scripts/mutants.sh                      # the whole crate (slow: ~2600 mutants)
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

# `--minimum-test-timeout` overrides cargo-mutants' 20s floor. Mutations that
# make a loop spin are detected by timing out, and they are common here (any
# comparison that ends an iteration can be flipped), so that floor dominates the
# run: the first full pass spent ~58 of its 120 minutes on 175 timeouts. The
# whole online-core suite runs in under a second, so 10s is a 10x margin even
# with four jobs competing. Raise it if a legitimate test is ever misreported as
# a timeout.
args=(--package online-core -j 4 --minimum-test-timeout 10)
for f in "$@"; do args+=(--file "$f"); done
cargo mutants "${args[@]}"
