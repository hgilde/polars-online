#!/usr/bin/env bash
# Mutation testing on online-core (docs/TESTING.md T-D4).
#
#   ./scripts/mutants.sh                      # the whole crate (slow: ~2600 mutants)
#   ./scripts/mutants.sh crates/online-core/src/clock.rs   # one file (~1 min)
#   ./scripts/mutants.sh --iterate            # only what the last run did not catch
#   ./scripts/mutants.sh --in-diff <(git diff main...)     # only code a diff touches
#
# Prefer the last two for follow-ups. A full pass is only worth it after a batch
# of feature work; `--iterate` answers "did the survivors close?" for a tenth of
# the cost, and `--in-diff` answers "is this branch covered?".
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
# Anything starting with `-` is a cargo-mutants flag; a bare word is a file to
# scope to. That keeps the common `./scripts/mutants.sh some/file.rs` working
# while allowing `--iterate` and `--in-diff` through.
while [ $# -gt 0 ]; do
    case "$1" in
        -*) args+=("$1"); shift; if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then args+=("$1"); shift; fi ;;
        *)  args+=(--file "$1"); shift ;;
    esac
done
cargo mutants "${args[@]}"
