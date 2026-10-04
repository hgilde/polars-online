#!/usr/bin/env bash
# Mutation testing on online-core (docs/TESTING.md T-D4).
#
#   ./scripts/mutants.sh                      # the whole crate (11,138 mutants on 2026-10-03; over a day at 4 jobs)
#   ./scripts/mutants.sh crates/online-core/src/clock.rs   # one file (85 mutants: minutes)
#   ./scripts/mutants.sh --iterate            # only what the last run did not catch
#   ./scripts/mutants.sh --in-diff <(git diff main...)     # only code a diff touches
#   python3 scripts/mutants_report.py mutants.out          # the survivors, less the equivalents
#
# Prefer the last two for follow-ups. A full pass is only worth it after a batch
# of feature work; `--iterate` answers "did the survivors close?" for a tenth of
# the cost, and `--in-diff` answers "is this branch covered?". CI runs both
# kinds itself (.github/workflows/mutants.yml): the changed lines on every push,
# failing on a survivor, and the whole crate weekly in forty-eight shards,
# reported, failing only when a shard did not finish (task 155).
# scripts/mutants_equivalent.toml lists the mutants no test can catch.
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

# Mutations that make a loop spin are detected by timing out, and they are
# common here (any comparison that ends an iteration can be flipped): the first
# full pass spent ~58 of its 120 minutes on 175 timeouts. A mutant's limit is a
# multiple of the baseline's test run, so two settings keep it short, as in CI
# (task 155). `--cargo-test-arg=--tests` skips the doctests, which the gate
# still runs: on 2026-10-03 the test binaries took 22 s here and the three
# doctests 39 s more. `--timeout-multiplier 3` stops a mutant at three times the
# baseline, where cargo-mutants' default is five: about a minute, where it was
# over five. Raise it on a busy machine, where a legitimate test can run that
# slow. `--minimum-test-timeout` lowers cargo-mutants' 20 s floor, which no
# longer binds.
args=(--package online-core -j 4 --minimum-test-timeout 10 --cargo-test-arg=--tests --timeout-multiplier 3)
# Anything starting with `-` is a cargo-mutants flag; a bare word is a file to
# scope to. That keeps the common `./scripts/mutants.sh some/file.rs` working
# while allowing `--iterate` and `--in-diff <(...)` through: a `-`-flag takes
# the following bare word as its value, which is what `--in-diff` needs.
#
# The corollary is that a value-less flag must not be followed by a bare file:
# `--iterate some/file.rs` hands the file to `--iterate` instead of `--file`.
# Scope with a bare file (`mutants.sh some/file.rs`) or pass `--iterate` alone;
# to combine, use `--file`: `mutants.sh --iterate --file some/file.rs` (review
# 2026-09-18).
while [ $# -gt 0 ]; do
    case "$1" in
        -*) args+=("$1"); shift; if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then args+=("$1"); shift; fi ;;
        *)  args+=(--file "$1"); shift ;;
    esac
done
cargo mutants "${args[@]}"
