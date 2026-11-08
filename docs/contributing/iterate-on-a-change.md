<!-- kind: how-to -->
# Iterate on a change

This how-to is for a contributor changing the library, who wants to run
the tests that cover a change while working on it, and the full suite
before a push.

**Run the tests for what you changed while you work, the essentials once
on each wave of two or more merged branches, and every test before each
push.** A task here is one change on its own branch. A wave is the set of
finished branches merged together for one test gate; a wave of one branch
skips the essentials and runs only the full suite before its push. Each
level reaches what the one before it cannot, and costs more. The
essentials and the extended tier are the suite's [two
tiers](../TESTING.md#two-tiers). The times are the project's development
Mac's, 14 cores shared with another user's jobs; the last two rows were
measured on 2026-10-11 at the load each gives.

| when | what runs | command | time |
|---|---|---|---|
| after each change, inside a task | the tests that cover the changed paths | `scripts/quick.sh` | seconds to a minute |
| at the end of a task | the same, with the mapped files' extended tests | `scripts/quick.sh --extended`, then the lints (step 3) | a minute or two |
| once on each wave of two or more finished branches, merged together; or each commit, working alone | every check, and the tests' essentials | `scripts/gate.sh` | 48 s in 8 processes at a load of 4; 111 s in 2 at a load of 20; 139 to 191 s in one |
| before every push, always | every check and every test, on the exact commit pushed | `scripts/gate.sh --extended` | 224 s in 6 processes at a load of 8; 487 to 501 s in one |

**Working alone, with no other branch in flight, run the essentials test
gate, `scripts/gate.sh`, on each commit in place of steps 1 to 3, and
merge onto `main` in place of step 4.** The quick loop and the wave are
for several branches built side by side, each in a worktree slot (a git
worktree kept between tasks, with its own builds), where one test gate
per branch would crowd the machine.

## Prerequisites

The steps need a checkout to work in and a machine with room to build:

- A checkout of the repository with its environment built (`uv sync`),
  on a branch for the change. A task works in a [worktree
  slot](use-a-worktree-slot.md), which rebuilds only what the change
  touches.
- The 1-minute load under about 20 (`uptime`), as [Keep the machine's load
  down](../TESTING.md#keep-the-machines-load-down) asks before any build.

## Steps

The steps run from the checkout's root, in the order a task reaches them:

1. **Run `scripts/quick.sh` after each change.** It runs the tests that
   cover the paths the branch changed, and its verdict reads `quick: PASS
   (the mapped tests only; the essentials gate runs on the merged wave)`. To
   see what it would run without running it, add `--dry-run`.
2. **At the end of the task, run `scripts/quick.sh --extended`.** It adds
   the mapped files' extended tests.
3. **Run the lints, the test gate's checks other than its tests.** Run
   `uv lock --check` first, before any `uv run`: `uv run` rewrites a stale
   `uv.lock`, so a later check would pass on the rewritten file. Then run
   `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets
   --locked -- -D warnings`, `uv run ruff format --check .`, `uv run ruff
   check .`, `uv run mypy` and `uv run --group docs sphinx-build -W -q -E
   docs/reference docs/_build/html`. Each passes with no finding.
   `scripts/quick.sh` is itself a `uv run`, so `git status` also shows
   whether steps 1 and 2 rewrote `uv.lock`.
4. **Commit the task, and leave `main` unmerged.** The branch joins the
   next wave ([Merge a wave of branches](merge-a-wave-of-branches.md)).
5. **Before every push, run `scripts/gate.sh --extended` on the exact
   commit pushed.** Its last line reads `gate: PASS (the extended tier: the
   full suite, the pre-push check)`.

To check the result, read the test gate's last line. `gate: PASS` names
the tier it ran. `gate: FAIL` names the steps that failed, with the last
25 lines of each.

[What `scripts/quick.sh` runs](../TESTING.md#what-scriptsquicksh-runs)
and [How the test gate runs](../TESTING.md#how-the-test-gate-runs)
describe the two scripts these steps call.
