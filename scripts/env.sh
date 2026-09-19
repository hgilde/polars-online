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

command -v cargo >/dev/null || echo "scripts/env.sh: cargo not found; install Rust from https://rustup.rs" >&2
command -v uv    >/dev/null || echo "scripts/env.sh: uv not found; see https://docs.astral.sh/uv/" >&2
