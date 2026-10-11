#!/usr/bin/env bash
# Put this project's toolchain on PATH:  source scripts/env.sh
#
# rustup installs cargo to ~/.cargo/bin and uv installs to ~/.local/bin; neither
# is guaranteed to be on a login shell's PATH (rustup's installer silently skips
# amending your profile if it is not writable). VS Code users get the same thing
# from .vscode/settings.json without sourcing anything.

for _dir in "$HOME/.cargo/bin" "$HOME/.local/bin"; do
    case ":$PATH:" in
        *":$_dir:"*) ;;                      # already present
        *) [ -d "$_dir" ] && PATH="$_dir:$PATH" ;;
    esac
done
unset _dir
export PATH

# online-py builds against pyo3's abi3-py312, so the Rust build needs a >= 3.12
# interpreter. `uv run cargo ...` handles this via VIRTUAL_ENV; for a bare
# `cargo ...` this makes the project venv the one pyo3 finds.
# `dirname` rather than `${BASH_SOURCE%/*}`: the latter strips nothing when
# the script is sourced by a bare name (`cd scripts && source env.sh`), so the
# venv probe would look inside `env.sh/..` and silently not set PYO3_PYTHON
# (review 2026-09-18). `dirname` returns `.` in that case, which is right.
_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$_dir/../.venv/bin/python" ]; then
    PYO3_PYTHON="$(cd "$_dir/.." && pwd)/.venv/bin/python"
    export PYO3_PYTHON
fi
unset _dir

# sccache as rustc's wrapper, only when asked: POLARS_ONLINE_SCCACHE=1. It
# caches each crate rustc compiles, keyed by its inputs, so a build that
# compiles a crate the same way again takes it from the cache. It changes no
# output: a hit returns the files rustc wrote for the same inputs.
#
# It is off by default because it pays only in a checkout it has seen
# before. Measured here on 2026-10-11: a cold build in a checkout whose
# path the cache already holds took 33 s for the Rust tests and 91 s for the
# release extension, against 66 s and 241 s without it. From a new path --
# a new worktree, the case it was meant for -- a third of the crates miss:
# every crate that reads a file its build script wrote names it by its
# absolute path in OUT_DIR, which differs per checkout, and the crates
# built on them (serde, proc-macro2 and through them Polars) miss too. The
# misses cost more than the hits save: 92 s and 519 s from a new path,
# slower than no cache at all.
#
# Local work only: it is never turned on where CI is set (every CI service
# sets it; tests/test_dev_loop.py holds that no workflow turns it on), nor
# where RUSTC_WRAPPER is already set, even to nothing. cargo rebuilds
# everything once when the wrapper changes. The cache is capped at
# SCCACHE_CACHE_SIZE, 10G unless set: the Rust tests' and the release
# extension's builds from one checkout took 0.8 GB, so several checkouts,
# toolchains and Polars versions fit; the oldest entries go first.
# scripts/clear_caches.sh empties it.
if [ -z "${CI:-}" ] && [ "${POLARS_ONLINE_SCCACHE:-0}" = 1 ] && [ -z "${RUSTC_WRAPPER+set}" ] \
    && command -v sccache >/dev/null; then
    export RUSTC_WRAPPER=sccache
    export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-10G}"
fi

command -v cargo >/dev/null || echo "scripts/env.sh: cargo not found; install Rust from https://rustup.rs" >&2
command -v uv    >/dev/null || echo "scripts/env.sh: uv not found; see https://docs.astral.sh/uv/" >&2
