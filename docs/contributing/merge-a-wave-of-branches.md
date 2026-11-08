<!-- kind: how-to -->
# Merge a wave of branches

This how-to is for the contributor who merges finished branches onto
`main`, and wants to land several of them behind one test gate.

**Several finished branches share one essentials test gate.** A wave is
the set of finished branches merged together for that test gate. Each
test gate builds and links the release extension with every core. Five at
once take 500 to 2,000 s each, against about 300 s for one alone, with its
release build. Merging each branch on its own would cost two things: one
test gate per branch, and a re-merge of `main` into every branch in flight
that touches the same files. So each branch ends with `scripts/quick.sh
--extended` and the lints ([Iterate on a change](iterate-on-a-change.md)),
and does not merge `main`.

**A wave of one branch skips `integration` and the essentials test gate.**
It has no merges to conflict, so it merges onto `main`, a fast-forward
where it can be. The full suite, which runs before every push anyway, then
runs once on that commit (`scripts/gate.sh --extended`). The
[steps](#steps) are for a wave of two branches or more.

## Prerequisites

The steps start from branches that are finished:

- Each branch of the wave has passed `scripts/quick.sh --extended` and
  the lints, and is committed.
- The 1-minute load under about 20 (`uptime`), as [Keep the machine's load
  down](../TESTING.md#keep-the-machines-load-down) asks before any build.

## Steps

The steps run in the main checkout:

1. **Cut an `integration` branch from `main`,** with `git switch -c
   integration main`.
2. **Merge the finished branches onto it in turn,** with `git merge
   <branch>` for each. Where a merge stops at a conflict confined to
   generated files, take either side with `git checkout --theirs
   <files>` (or `--ours`), then `git add <files>` and `git commit` to finish that
   merge before the next.
3. **After the last merge, regenerate the conflicted generated files
   once.** The state fixtures regenerate with
   `PRINT_STATE_FIXTURES=1` ([What each change does to a saved
   state](add-an-output-or-a-parameter.md#what-each-change-does-to-a-saved-state),
   step 4), and the API snapshot with `UPDATE_API_SURFACE=1` ([Change the
   public API](change-the-api.md#steps), step 4). The golden values only
   print: `PRINT_GOLDEN=1 cargo test -p online-core --test golden --
   --nocapture` prints Rust's as lines `GOLDEN <kind>: &[...];`. Copy each
   printed list into the constant that the test's `check("<kind>", …)`
   call passes, under the `// ---
   generated` marker in `crates/online-core/tests/golden.rs`. `uv run python
   tests/test_golden_pipeline.py` prints Python's, to paste over `GOLDEN`
   and `SWITCHED_GOLDEN` in that file. A golden value that moved is a
   regression until a branch's change explains it, so find out why before
   you copy it. Commit the regenerated files.
4. **Run one essentials test gate, `scripts/gate.sh`, on `integration`.**
   Its last line reads `gate: PASS` or `gate: FAIL`.
5. **On PASS, fast-forward `main` to `integration`,** with `git switch
   main` and `git merge --ff-only integration`, then delete the branch
   with `git branch -d integration`.
6. **On a failure, find the branch.** The failure names its tests.
   `scripts/slot.sh overlap` and `git log` tie them to the branch that
   touched their files. Where that is not enough, a bisection over the
   wave's merges finds it in about log₂ of the wave's size in test gates.
7. **Send that branch back to its task, and land the rest of the wave
   without it.** Delete `integration` with `git switch main` and `git
   branch -D integration`, then repeat steps 1 to 5 with the rest.
8. **Before the push, run `scripts/gate.sh --extended` on `main`,** the
   exact commit pushed ([Iterate on a change](iterate-on-a-change.md),
   step 5).
9. **Push `main`,** with `git push origin main`, once that run passes.
10. **Give back each merged branch's worktree slot,** with `scripts/slot.sh
    give <n>` ([Use a worktree slot](use-a-worktree-slot.md), step 5).

To check the result, run `git log --oneline main`. Its newest commits
are the wave's merges, and the commit of any files regenerated in
step 3.
