#!/usr/bin/env bash
# Long-lived worktree slots, each with its own warm target/ and .venv, so a
# task rebuilds only the workspace crates it changes, not Polars and its
# dependencies (docs/TESTING.md, "Worktree slots").
#
#   scripts/slot.sh take <branch> [<base>]   a free slot, on a new <branch> from <base> (default main);
#                                            prints the slot's path
#   scripts/slot.sh give <slot>              free it (a number or its path); refuses uncommitted work
#   scripts/slot.sh list                     every slot, and who holds it
#   scripts/slot.sh overlap [<branch>...]    the files two taken slots both touch, and the files
#                                            main changed under each since its base: each is a
#                                            merge conflict, and a gate run, waiting to happen.
#                                            Branches named after the slots count as slots too
#                                            (a branch about to be merged or dispatched).
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
# (default 4). A slot stays warm between tasks, so there are more slots than
# agents building: at most two build at once (docs/TESTING.md, "At most two
# agents"). Who holds a
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
    echo "usage: scripts/slot.sh take <branch> [<base>] | give <slot> | list | overlap [<branch>...]" >&2
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
    overlap)
        # What each branch touches: its commits since its merge base with
        # main, plus a slot's uncommitted files. Every file two branches both
        # touch conflicts, or at least makes the later one merge main again
        # and re-gate; so does a file main changed since the branch's base.
        tmp="$(mktemp -d)"
        trap 'rm -rf "$tmp"' EXIT
        names=()
        for n in $(seq 1 "$max"); do
            [ -e "$locks/$n" ] || continue
            dir="$(slot_dir "$n")"
            [ -e "$dir/.git" ] || continue
            name="slot$n:$(awk '{print $1}' "$locks/$n")"
            base="$(git -C "$dir" merge-base main HEAD)"
            { git -C "$dir" diff --name-only "$base" HEAD
              git -C "$dir" status --porcelain | cut -c4-; } | sort -u > "$tmp/$n"
            git -C "$main" diff --name-only "$base" main | sort -u > "$tmp/$n.main"
            names+=("$n=$name")
        done
        shift
        for b in "$@"; do
            git -C "$main" rev-parse --verify --quiet "$b^{commit}" >/dev/null \
                || { echo "slot: no branch $b" >&2; exit 2; }
            base="$(git -C "$main" merge-base main "$b")"
            key="b$(printf '%s' "$b" | tr -c 'A-Za-z0-9' '_')"
            git -C "$main" diff --name-only "$base" "$b" | sort -u > "$tmp/$key"
            git -C "$main" diff --name-only "$base" main | sort -u > "$tmp/$key.main"
            names+=("$key=$b")
        done
        found=0
        for entry in "${names[@]}"; do
            k="${entry%%=*}"
            shared="$(comm -12 "$tmp/$k" "$tmp/$k.main")"
            if [ -n "$shared" ]; then
                found=1
                echo "${entry#*=} vs main (changed since its base):"
                echo "$shared" | sed 's/^/    /'
            fi
        done
        i=0
        for a in "${names[@]}"; do
            i=$((i + 1))
            j=0
            for b in "${names[@]}"; do
                j=$((j + 1))
                [ "$j" -gt "$i" ] || continue
                shared="$(comm -12 "$tmp/${a%%=*}" "$tmp/${b%%=*}")"
                if [ -n "$shared" ]; then
                    found=1
                    echo "${a#*=} vs ${b#*=} ($(echo "$shared" | wc -l | tr -d ' ') files):"
                    echo "$shared" | sed 's/^/    /'
                fi
            done
        done
        [ "$found" = 1 ] || echo "no overlap"
        ;;
    *) usage ;;
esac
