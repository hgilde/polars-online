#!/usr/bin/env bash
# Clear the build caches this checkout fills, and say what each freed
# (docs/TESTING.md, "Caches and when to clear them").
#
#   scripts/clear_caches.sh              the build caches: target/ and sccache's cache
#   scripts/clear_caches.sh --dry-run    what it would clear, and the size of each
#   scripts/clear_caches.sh --worktrees  also target/ in every other worktree of this repo
#   scripts/clear_caches.sh --test-data  also .cache/ (downloads) and .hypothesis/
#   scripts/clear_caches.sh --uv         also `uv cache prune` (uv's own cache, outside the repo)
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
test_data=
uv_prune=
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry=1 ;;
        --worktrees) worktrees=1 ;;
        --test-data) test_data=1 ;;
        --uv) uv_prune=1 ;;
        *) echo "usage: scripts/clear_caches.sh [--dry-run] [--worktrees] [--test-data] [--uv]" >&2; exit 2 ;;
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
if [ -n "$worktrees" ]; then
    while IFS= read -r line; do
        case "$line" in
            "worktree "*) wt="${line#worktree }"; [ "$wt" != "$repo" ] && targets+=("$wt/target") ;;
        esac
    done < <(git worktree list --porcelain)
fi
if [ -n "$test_data" ]; then
    targets+=("$repo/.cache" "$repo/.hypothesis")
fi

size_kb() { du -sk "$1" 2>/dev/null | cut -f1; }
human() { awk -v k="$1" 'BEGIN { if (k >= 1048576) printf "%.1f GB", k / 1048576; else printf "%.0f MB", k / 1024 }'; }

if [ -z "$dry" ] && command -v sccache >/dev/null; then
    sccache --stop-server >/dev/null 2>&1 || true
fi

total=0
for path in "${targets[@]}"; do
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
