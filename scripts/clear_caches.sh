#!/usr/bin/env bash
# Clear the build caches this checkout fills, and say what each freed
# (docs/contributing/clear-the-caches.md).
#
#   scripts/clear_caches.sh              the build caches: target/ and sccache's cache
#   scripts/clear_caches.sh --dry-run    what it would clear, and the size of each
#   scripts/clear_caches.sh --worktrees  also target/ in every other worktree of this repo, but
#                                        not the worktree slots' (scripts/slot.sh), nor the main
#                                        checkout's when run from another worktree
#   scripts/clear_caches.sh --slots      also the slots' target/, which keeping them warm is for
#   scripts/clear_caches.sh --test-data  also .cache/ (downloads) and .hypothesis/
#   scripts/clear_caches.sh --uv         also `uv cache prune` (uv's own cache, outside the repo)
#   scripts/clear_caches.sh --stale [N]  instead of removing target/ whole, remove only what no build has
#                                        read for N days (default 7) from each target/ in the list: the
#                                        superseded test binaries and rlibs in deps/ and the old
#                                        incremental/ sessions; sccache and the other flags' caches stay
#
# By default it removes nothing but target/ (debug, release, dev-wheel,
# nextest's files; what `cargo clean` removes) and sccache's cache
# directory, after stopping the sccache server so it is not writing there.
# Every one of them is rebuilt by the next build, at the cost the guide
# gives. The test data under .cache/ is downloaded again only on a network,
# so it goes only with --test-data. Nothing else outside the repo is
# touched unless a flag names it.
set -uo pipefail
cd "$(dirname "$0")/.."
repo="$(pwd)"

dry=
worktrees=
slots=
test_data=
uv_prune=
stale=
days=7
want_days=
for arg in "$@"; do
    if [ -n "$want_days" ]; then
        want_days=
        case "$arg" in
            ''|*[!0-9]*) ;;
            *) days="$arg"; continue ;;
        esac
    fi
    case "$arg" in
        --stale) stale=1; want_days=1 ;;
        --dry-run) dry=1 ;;
        --worktrees) worktrees=1 ;;
        --slots) slots=1 ;;
        --test-data) test_data=1 ;;
        --uv) uv_prune=1 ;;
        *) echo "usage: scripts/clear_caches.sh [--dry-run] [--worktrees] [--slots] [--test-data] [--uv] [--stale [DAYS]]" >&2; exit 2 ;;
    esac
done

# sccache's cache: SCCACHE_DIR, or its default for the platform.
if [ -n "${SCCACHE_DIR:-}" ]; then
    sccache_dir="$SCCACHE_DIR"
elif [ "$(uname -s)" = Darwin ]; then
    sccache_dir="$HOME/Library/Caches/Mozilla.sccache"
else
    sccache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/sccache"
fi

# A cache directory set to something else by mistake ($HOME, /) is refused,
# not emptied: its last part must name sccache.
case "$(basename "$sccache_dir" | tr '[:upper:]' '[:lower:]')" in
    *sccache*) ;;
    *) echo "clear_caches: refusing $sccache_dir: its name does not say sccache" >&2; exit 2 ;;
esac

targets=("$repo/target" "$sccache_dir")
if [ -n "$worktrees" ] || [ -n "$slots" ]; then
    # The slots belong to the main checkout, as scripts/slot.sh makes them,
    # wherever this runs from: matched against this checkout, a run from
    # inside a slot took every other slot for an ordinary worktree.
    main="$(dirname "$(cd "$(git rev-parse --git-common-dir)" && pwd)")"
    while IFS= read -r line; do
        case "$line" in
            "worktree "*)
                wt="${line#worktree }"
                [ "$wt" = "$repo" ] && continue
                # The main checkout's build is where the coordinator's gate
                # runs: only a run from there clears it (task 269, the
                # second look after review round 8).
                [ "$wt" = "$main" ] && continue
                # A slot is kept warm on purpose: it goes only with --slots.
                case "$wt" in
                    "$main/.claude/slots/"*) [ -n "$slots" ] && targets+=("$wt/target") ;;
                    *) [ -n "$worktrees" ] && targets+=("$wt/target") ;;
                esac
                ;;
        esac
    done < <(git worktree list --porcelain)
fi
if [ -n "$test_data" ]; then
    targets+=("$repo/.cache" "$repo/.hypothesis")
fi

size_kb() { du -sk "$1" 2>/dev/null | cut -f1; }
human() { awk -v k="$1" 'BEGIN { if (k >= 1048576) printf "%.1f GB", k / 1048576; else printf "%.0f MB", k / 1024 }'; }

if [ -z "$dry" ] && [ -z "$stale" ] && command -v sccache >/dev/null; then
    sccache --stop-server >/dev/null 2>&1 || true
fi

# --stale: a file a build has read since the cutoff is in use, whatever its
# mtime says (cargo reads each artifact it links or fingerprints, so a
# current Polars rlib is read by every build and survives); a superseded
# binary nothing reads goes. Cargo rebuilds any file it finds missing.
prune_stale() {
    local path=$1 before after
    before=$(size_kb "$path")
    if [ -z "$dry" ]; then
        find "$path" -type f \( -path '*/deps/*' -o -path '*/incremental/*' \) -atime +"$days" -delete 2>/dev/null
        find "$path" -type d -empty -path '*/incremental/*' -delete 2>/dev/null
        after=$(size_kb "$path")
    else
        after=$((before - $(find "$path" -type f \( -path '*/deps/*' -o -path '*/incremental/*' \) -atime +"$days" -exec du -k {} + 2>/dev/null | awk '{s += $1} END {print s + 0}')))
    fi
    echo $((before - after))
}

total=0
for path in "${targets[@]}"; do
    if [ -n "$stale" ]; then
        case "$path" in */target) ;; *) continue ;; esac
        [ -e "$path" ] || { echo "absent      $path"; continue; }
        kb=$(prune_stale "$path")
        total=$((total + kb))
        if [ -n "$dry" ]; then
            printf 'would free  %-9s %s (unread for %s days)\n' "$(human "$kb")" "$path" "$days"
        else
            printf 'freed       %-9s %s (unread for %s days)\n' "$(human "$kb")" "$path" "$days"
        fi
        continue
    fi
    if [ ! -e "$path" ]; then
        echo "absent      $path"
        continue
    fi
    kb=$(size_kb "$path")
    total=$((total + kb))
    if [ -n "$dry" ]; then
        printf 'would free  %-9s %s\n' "$(human "$kb")" "$path"
    else
        rm -rf "$path"
        printf 'freed       %-9s %s\n' "$(human "$kb")" "$path"
    fi
done
if [ -n "$uv_prune" ]; then
    if [ -n "$dry" ]; then
        echo "would run   uv cache prune"
    else
        uv cache prune
    fi
fi
if [ -n "$dry" ]; then
    echo "clear_caches: would free $(human "$total") (dry run: nothing removed)"
else
    echo "clear_caches: freed $(human "$total")"
fi
