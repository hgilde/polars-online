<!-- kind: how-to -->
# Use a worktree slot

This how-to is for a contributor starting a task on the project's
development Mac, who wants to take a warm worktree slot, work in it, and
give it back once the branch is merged.

**A task works in a long-lived worktree slot, so it rebuilds only what it
changes.** A fresh worktree starts with an empty `target/` and an empty
`.venv/`, and builds Polars and every dependency before its first test. A
worktree slot is a worktree kept between tasks, with its own `target/`
and `.venv/`. Taking it for a new task moves it to the task's branch with
`git checkout -B`, and cargo's fingerprints then rebuild only the
workspace crates the move changed.

## Prerequisites

A worktree slot needs the main checkout and room for its builds:

- The main checkout of the repository. The worktree slots live under its
  `.claude/slots/<n>`, four by default (`POLARS_ONLINE_SLOTS` sets the
  number).
- A new name for the task's branch. `take` resets an existing branch of
  that name to its base, so the branch's own commits leave it.
- Room on the disk for a worktree slot's builds, about 28 GB, and 0.9 GB
  of environment ([Clear the caches](clear-the-caches.md) frees it).

## Steps

The steps follow a task from its start to its merge:

1. **Check what the taken worktree slots change,** with `git -C
   .claude/slots/<n> diff --name-only main...HEAD` for each taken one
   (`scripts/slot.sh list` names them), and `git -C .claude/slots/<n>
   status --short` for its uncommitted files. Start the task elsewhere,
   or later, if the files it will touch are among them. `scripts/slot.sh
   overlap` cannot answer this: it lists only the files two taken
   worktree slots share, or one shares with `main`, and the new task has
   no worktree slot yet.
2. **Take a worktree slot,** with `scripts/slot.sh take <branch> main`. It
   moves a free worktree slot onto a new branch cut from `main`, and
   prints its path.
3. **Work only in that path.** Build, test and commit there, as [Iterate
   on a change](iterate-on-a-change.md) describes.
4. **Commit everything before the branch is merged.** `give` refuses a
   worktree slot with uncommitted work.
5. **After the branch is merged, give the worktree slot back,** with
   `scripts/slot.sh give <n>`, where `<n>` is its number or its path. It
   detaches the worktree slot, so the task's branch can be deleted.

To check the result, run `scripts/slot.sh list`. It lists every worktree
slot and who holds it, and the one you gave back shows as free.

```
scripts/slot.sh take my-change main       # prints the worktree slot's path
scripts/slot.sh list                      # every worktree slot, and who holds it
scripts/slot.sh give 1                    # after the branch is merged
```

## What a warm worktree slot saves

**A warm worktree slot rebuilds the Rust tests in 23 s and the inner
loop's extension in 6 s, where one built from nothing takes 79 s and
145 s.** Measured on 2026-10-13, at a load of 10 to 20: a worktree slot
built at one commit and moved to a `main` 40 files and 5,120 lines later,
against the same worktree slot from nothing.

| build | from nothing | a warm worktree slot, moved |
|---|---:|---:|
| the Rust tests (`cargo test --no-run`) | 79 s | 23 s |
| the environment and its editable install (`uv sync`) | 259 s | 0 s |
| the inner loop's extension (`dev-wheel`) | 145 s | 6 s |
| the test gate's extension (`release`) | built by `uv sync` above | 198 s |

The `release` extension gains least. Thin LTO over one codegen unit
relinks the whole extension whenever a workspace crate changes. Only the
test gate builds it: once a wave, or once a commit when working alone, and
again before each push. The
inner loop builds `dev-wheel`.

## What `take` and `give` refuse, and why

**Each worktree slot has its own `target/`, never a shared one.** cargo
hashes a package's path relative to the workspace root. So two checkouts
building into one `target/` overwrite each other's artifacts, and a test
gate can then test another checkout's binary. Each worktree slot holds
about 28 GB of builds and 0.9 GB of environment. It stays warm between
tasks whether or not a contributor is building in it, so there are more
worktree slots than builds running. At most two build at once, three
while the load is low, as [Keep the machine's load
down](../TESTING.md#keep-the-machines-load-down) says.

| command | refuses | because |
|---|---|---|
| `take` | a worktree slot with uncommitted work | the work would be lost with the move |
| `take` | a branch another worktree has checked out, naming that worktree | `git checkout -B` would move the worktree slot onto it, and a commit in either worktree would leave the other's files behind its branch |
| `take` | when every worktree slot is taken | there is none free; `scripts/slot.sh list` shows who holds each |
| `give` | a worktree slot with uncommitted work | the work would be lost with the worktree slot |
| `give` | a worktree slot that is not taken | there is nothing to give back |

Who holds a worktree slot is a file in the repository's git directory, so
the worktree's `git status` stays clean. `scripts/clear_caches.sh
--worktrees` keeps the worktree slots' builds, and `--slots` clears them
too.

## Check what the worktree slots share

**Check what the worktree slots share before merging a branch.** A file two branches both change costs the later one a merge of
`main`, often a conflict, and a second test gate. So does a file `main`
changed since a branch's base. `scripts/slot.sh overlap` lists both cases
for every taken worktree slot. It also checks each existing branch named
after it, such as one about to be merged:

```
scripts/slot.sh overlap                   # the taken worktree slots against each other and main
scripts/slot.sh overlap my-change         # the same, with an existing branch not in a worktree slot
```

Run one after the other the branches that share a file that tooling
regenerates (the state fixtures, the golden values), or a file most
changes touch (`stream.rs`, the shared test cases). Each such pair run
side by side costs the later branch a re-merge and another test gate.
Branches over separate files merge without one.
