#!/usr/bin/env bash
# Long-lived worktree slots, each with its own warm target/ and .venv, so a
# task rebuilds only the workspace crates it changes, not Polars and its
# dependencies (docs/TESTING.md, "Worktree slots").
#
#   scripts/slot.sh take <branch> [<base>]   a free slot, on a new <branch> from <base> (default main);
#                                            prints the slot's path
#   scripts/slot.sh give <slot>              free it (a number or its path); refuses uncommitted work
#   scripts/slot.sh list                     every slot, and who holds it
#
# A fresh worktree starts from an empty target/: measured, 66 s for the Rust
# tests, 241 s for the release wheel and 130 s for the dev wheel before any
# test runs, where a warm one rebuilds after an edit in 4.5-37 s. A slot is
# a worktree kept between tasks: `take` moves it to the new branch with
# `git checkout -B` (refusing a branch another worktree holds), and cargo's
# fingerprints then rebuild only what the move changed. Each slot keeps its
# own target/, never a shared one: cargo hashes package paths relative to
# the workspace root, so two checkouts in one target/ overwrite each other's
# artifacts.
#
# Slots live under .claude/slots/<n>, at most POLARS_ONLINE_SLOTS of them
# (default 4, the number of agents this machine runs at once). Who holds a
# slot is a file in the repository's common git directory, so it never
# shows in a slot's `git status`. `clear_caches.sh --worktrees` keeps the
# slots' target/ unless it is given --slots.
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
common="$(cd "$repo" && cd "$(git rev-parse --git-common-dir)" && pwd)"
# The slots belong to the main checkout, wherever this is run from.
main="$(dirname "$common")"
locks="$common/slots"
max="${POLARS_ONLINE_SLOTS:-4}"

usage() {
    echo "usage: scripts/slot.sh take <branch> [<base>] | give <slot> | list" >&2
    exit 2
}

slot_dir() { echo "$main/.claude/slots/$1"; }

slot_of() {
    # A slot named by its number or by its path.
    case "$1" in
        '' ) usage ;;
        *[!0-9]*) basename "$1" ;;
        *) echo "$1" ;;
    esac
}

cmd="${1:-}"
case "$cmd" in
    take)
        branch="${2:-}"
        base="${3:-main}"
        [ -n "$branch" ] || usage
        git -C "$main" rev-parse --verify --quiet "$base^{commit}" >/dev/null \
            || { echo "slot: no commit $base" >&2; exit 2; }
        mkdir -p "$locks"
        for n in $(seq 1 "$max"); do
            # The lock is taken by creating it, so two takers cannot both win.
            if ( set -o noclobber; printf '%s %s\n' "$branch" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$locks/$n" ) 2>/dev/null; then
                dir="$(slot_dir "$n")"
                # `checkout -B` does not refuse a branch another worktree has
                # checked out, as a plain checkout does; two worktrees on one
                # branch leave one's tree behind the HEAD the other commits.
                holder="$(git -C "$main" worktree list --porcelain | awk -v ref="branch refs/heads/$branch" '
                    /^worktree / { wt = substr($0, 10) }
                    $0 == ref && !found { print wt; found = 1 }')"
                if [ -n "$holder" ] && { [ ! -e "$dir/.git" ] || [ "$(cd "$holder" && pwd -P)" != "$(cd "$dir" && pwd -P)" ]; }; then
                    rm -f "$locks/$n"
                    echo "slot: $branch is checked out at $holder; take a new branch, or free that one there first" >&2
                    exit 1
                fi
                if [ ! -e "$dir/.git" ]; then
                    mkdir -p "$(dirname "$dir")"
                    git -C "$main" worktree add --quiet --detach "$dir" "$base" >&2
                fi
                if [ -n "$(git -C "$dir" status --porcelain)" ]; then
                    rm -f "$locks/$n"
                    echo "slot: $dir has uncommitted work; finish or discard it there first" >&2
                    exit 1
                fi
                git -C "$dir" checkout --quiet -B "$branch" "$base" >&2
                echo "$dir"
                exit 0
            fi
        done
        echo "slot: all $max slots are taken (scripts/slot.sh list)" >&2
        exit 1
        ;;
    give)
        n="$(slot_of "${2:-}")"
        dir="$(slot_dir "$n")"
        [ -e "$locks/$n" ] || { echo "slot: slot $n is not taken" >&2; exit 1; }
        if [ -e "$dir/.git" ] && [ -n "$(git -C "$dir" status --porcelain)" ]; then
            echo "slot: $dir has uncommitted work; commit or discard it before giving the slot back" >&2
            exit 1
        fi
        # Detach, so the task's branch can be merged and deleted from main.
        [ -e "$dir/.git" ] && git -C "$dir" checkout --quiet --detach
        rm -f "$locks/$n"
        echo "slot $n free"
        ;;
    list)
        for n in $(seq 1 "$max"); do
            dir="$(slot_dir "$n")"
            if [ -e "$locks/$n" ]; then
                echo "$n taken by $(cat "$locks/$n")  $dir"
            elif [ -e "$dir/.git" ]; then
                echo "$n free  $dir"
            else
                echo "$n not made yet"
            fi
        done
        ;;
    *) usage ;;
esac
