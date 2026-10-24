#!/usr/bin/env bash
# Run a command with its memory capped as the kernel counts it, in a
# transient systemd scope, on a CI runner (mutants.yml):
#
#   bash scripts/memory_scope.sh 13G -- cargo mutants ...
#
# The scope's cgroup has `MemoryMax` (what its processes have resident, page
# cache included, which the kernel reclaims first) and no swap. A command
# that goes past it has its largest process killed by the kernel's OOM
# killer -- a runaway mutant's test, which its test runner then counts as a
# failed test, so the mutant is caught -- and `OOMPolicy=continue` keeps the
# rest of the scope running. Everything outside the scope, the runner's own
# agent among it, keeps the memory left over.
#
# Why not `prlimit --data` on each test process, as before: RLIMIT_DATA
# counts a process's private writable *virtual* mappings, which glibc keeps
# after a free (its arenas, the heap it retains), not what is resident. The
# suite's resident peak was measured on macOS at 497 MiB for online-core's
# largest test binary, and a cap of 6 GiB of "data" still killed it on Linux
# (`memory allocation of 545259520 bytes failed` in the baseline of 4 of the
# 32 shards of run 38205650365), depending on which tests shared a process.
#
# The scope is made as root (`sudo systemd-run --scope`, the runners allow
# it), and the command runs as the calling user again (`setpriv`), with the
# caller's whole environment: sudo resets it, and `sudo -E`, which kept it,
# is ignored by ubuntu-26.04's sudo-rs ("preserving the entire environment
# is not supported"). There the mutants ran without the workflow's
# CARGO_INCREMENTAL=0 and without any TMPDIR it set. So the environment
# goes through a file, NUL-separated (`env -0`), which the command's side
# exports again before it runs. Its exit status is written
# to a file from inside the scope and returned here, whatever systemd-run
# itself returns. Linux with systemd and cgroup v2 only: GitHub's Ubuntu
# runners.
set -uo pipefail

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
    echo "usage: scripts/memory_scope.sh MEMORY_MAX -- COMMAND..." >&2
    exit 2
fi
max="$1"
shift 2

if [ -n "${MEMORY_SCOPE_STATUS:-}" ]; then
    # Inside the scope, as the calling user: the caller's environment back,
    # then run, and leave the status. A name bash will not take (readonly,
    # or not an identifier) is passed over.
    if [ -n "${MEMORY_SCOPE_ENV:-}" ]; then
        while IFS= read -r -d '' _scope_kv; do
            _scope_name="${_scope_kv%%=*}"
            case "$_scope_name" in
                MEMORY_SCOPE_* | BASH* | SHELLOPTS | UID | EUID | PPID) continue ;;
            esac
            [[ "$_scope_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
            export "$_scope_kv" 2>/dev/null || true
        done < "$MEMORY_SCOPE_ENV"
        rm -f "$MEMORY_SCOPE_ENV"
        unset MEMORY_SCOPE_ENV _scope_kv _scope_name
    fi
    "$@"
    status=$?
    echo "$status" > "$MEMORY_SCOPE_STATUS"
    exit "$status"
fi

status_file="$(mktemp)"
rm -f "$status_file"
env_file="$(mktemp)"
env -0 > "$env_file"
sudo systemd-run --scope --quiet \
    -p MemoryMax="$max" -p MemorySwapMax=0 -p OOMPolicy=continue \
    -- setpriv --reuid="$(id -u)" --regid="$(id -g)" --init-groups \
    env PATH="$PATH" HOME="$HOME" MEMORY_SCOPE_STATUS="$status_file" \
    MEMORY_SCOPE_ENV="$env_file" \
    bash "$0" "$max" -- "$@"
launched=$?
rm -f "$env_file"
if [ ! -s "$status_file" ]; then
    echo "::error::memory_scope.sh: the command never finished inside the scope (systemd-run exited $launched)" >&2
    exit 1
fi
status="$(cat "$status_file")"
rm -f "$status_file"
exit "$status"
