<!-- kind: how-to -->
# Clear the caches

This how-to is for a contributor on the project's development Mac who
wants to free disk space by clearing the build caches, and to know when
to do it.

**Clear the build caches after each push wave, once its branches and
worktrees are removed, or when free disk falls below 100 GB.** Every cache can be deleted while no build or test is using it.
The next build or run refills what it needs, at the cost [What each cache
holds](#what-each-cache-holds) gives.

## Prerequisites

Clear the caches only once nothing still needs their builds:

- The main checkout of the repository, where you run the script.
- The push wave's branches merged and pushed, and their worktrees removed
  with `git worktree remove`, so that no build you still need is cleared.

## Steps

Run each step from the main checkout:

1. **See what would go,** with `scripts/clear_caches.sh --worktrees
   --dry-run`. It prints what it would remove and
   the size of each, and removes nothing.
2. **Clear the caches,** with `scripts/clear_caches.sh --worktrees`. It
   removes this checkout's `target/`, every other
   worktree's `target/` but the worktree slots', and sccache's cache, and prints
   what each freed.
3. **Add a flag only for a cache you mean to lose.** `--slots` also clears
   the worktree slots' `target/`, so the next task in each worktree slot builds
   from nothing. `--test-data` also clears `.cache/` and `.hypothesis/`,
   and `--uv` runs `uv cache prune`.

To check the result, read the sizes the script printed, and the free disk
it left with `df -h .`.

## What the script clears

**`scripts/clear_caches.sh` clears the build caches and prints what
each freed.** By default it removes this checkout's `target/` and sccache's
cache, after stopping the sccache server. `--worktrees` adds every other
worktree's `target/`, with two left alone. It keeps the worktree slots', which it
knows by the main checkout's `.claude/slots/` wherever it runs from. And
when it runs from another worktree, it keeps the main checkout's, so a
contributor in another worktree cannot clear the build a test gate in
the main checkout is using.

| flag | adds |
|---|---|
| `--worktrees` | every other worktree's `target/`, but the worktree slots' and, from another worktree, the main checkout's |
| `--slots` | the worktree slots' `target/` |
| `--test-data` | `.cache/` and `.hypothesis/` |
| `--uv` | `uv cache prune` |
| `--dry-run` | prints what it would remove, and removes nothing |

It touches nothing outside the repository and sccache's own directory
unless a flag names it (`tests/test_dev_loop.py`).

## What each cache holds

**The builds and the tests fill these directories, and each can be
deleted while no build or test is using it.** The next build or run
refills what it needs, at the cost the last column gives. The sizes were
measured on the development Mac on 2026-10-11.

| what | where | filled by | size | the next build after deleting it |
|---|---|---|---|---|
| the Rust tests' build | `target/debug/` in each checkout | `cargo test`, `cargo nextest`, `cargo clippy` | 21 to 23 GB, most of it incremental state | 66 s to build the tests again |
| the test gate's extension | `target/release/` | `maturin develop --release` | 1.5 GB | 241 s |
| the inner loop's extension | `target/dev-wheel/` | `maturin develop --profile dev-wheel` | 2.3 to 2.4 GB | 130 s |
| each other worktree's builds | `target/` under `.claude/worktrees/<name>/` | the three builds above | up to 27 GB each | removed with the worktree, by `git worktree remove` |
| each worktree slot's builds | `target/` under `.claude/slots/<n>/` | the three builds above, kept between tasks | about 28 GB each, four by default | kept by `clear_caches.sh --worktrees`; `--slots` clears them, and the next task in that worktree slot builds from nothing |
| sccache's cache, when turned on | `~/Library/Caches/Mozilla.sccache` on macOS | builds with `POLARS_ONLINE_SCCACHE=1` | 0.8 GB per checkout path, capped at 10 GB (`SCCACHE_CACHE_SIZE`), oldest out first | a build from nothing |
| downloaded test data | `.cache/` in each checkout | `tests/data.py`, `scripts/compare_release.py` | 1.1 GB after a full run in a new worktree; 7.1 GB in the main checkout, with the experiments' data | the downloads again, about a minute on a fast line; offline, the tests that need them skip |
| pytest's temporary directories | `pytest-of-<user>/` in the system's temporary directory | every test's `tmp_path` | about 2 GB for each full run, since the tree-model tests write a 2 GB source, and the last three runs are kept | nothing; pytest keeps three and deletes the rest itself |
| Hypothesis's examples | `.hypothesis/` | every property test | under 3 MB | the property tests lose the failing cases they replay first |
| uv's cache | `~/.cache/uv`, shared by every project | `uv sync`, `uv run` | 23 GB | the next sync downloads what it lacks; `uv cache prune` removes only unused entries |
| the Python environment | `.venv/` in each checkout | `uv sync` | 0.9 GB | `uv sync`, seconds from uv's cache |

pytest-xdist and cargo-nextest leave nothing of their own: xdist's workers
use pytest's temporary directories, under the same base as a serial run,
and nextest writes no files unless asked for a report.
