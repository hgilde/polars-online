<!-- kind: how-to -->
# Use a worktree slot

This how-to is for a contributor starting a task on this machine, who
wants to take a warm worktree slot, work in it, and give it back once the
branch is merged.

**A task works in a long-lived worktree slot, so it rebuilds only what it
changes.** A fresh worktree starts with an empty `target/` and an empty
`.venv/`, and builds Polars and every dependency before its first test. A
slot is a worktree kept between tasks, with its own `target/` and
`.venv/`. Taking it for a new task moves it to the task's branch with `git
checkout -B`, and cargo's fingerprints then rebuild only the workspace
crates the move changed.

## Prerequisites

A slot needs the main checkout and room for its builds:

- The main checkout of the repository. The slots live under its
  `.claude/slots/<n>`, at most four of them (`POLARS_ONLINE_SLOTS`).
- A name for the task's branch, which no other worktree has checked out.
- Room on the disk for a slot's builds, about 28 GB, and 0.9 GB of
  environment ([Clear the caches](clear-the-caches.md) frees it).

## Steps

The steps follow a task from its start to its merge:

1. **Check what the branch would share with the taken slots,** with
   `scripts/slot.sh overlap <branch>`. It lists the files two taken slots
   both change, and the files `main` changed under each since its base.
2. **Take a slot,** with `scripts/slot.sh take <branch> main`. It moves a
   free slot onto a new branch cut from `main`, and prints the slot's
   path.
3. **Work only in that path.** Build, test and commit there, as [Iterate
   on a change](iterate-on-a-change.md) describes.
4. **Commit everything before the branch is merged.** `give` refuses a
   slot with uncommitted work.
5. **After the branch is merged, give the slot back,** with
   `scripts/slot.sh give <n>`, where `<n>` is the slot's number or its
   path. It detaches the slot, so the task's branch can be merged and
   deleted from `main`.

To check the result, run `scripts/slot.sh list`. It lists every slot and
who holds it, and the slot you gave back shows as free.

```
scripts/slot.sh take task-252-core main   # prints the slot's path
scripts/slot.sh list                      # every slot, and who holds it
scripts/slot.sh give 1                    # after the branch is merged
```

## What a warm slot saves

**A warm slot rebuilds the Rust tests in 23 s and the inner loop's
extension in 6 s, where a slot built from nothing takes 79 s and 145 s.**
Measured on 2026-10-13, a slot built at one commit and moved to a `main`
40 files and 5,120 lines later, against the same slot from nothing, at a
load of 10 to 20:

| build | from nothing | a warm slot, moved |
|---|---:|---:|
| the Rust tests (`cargo test --no-run`) | 79 s | 23 s |
| the environment and its editable install (`uv sync`) | 259 s | 0 s |
| the inner loop's extension (`dev-wheel`) | 145 s | 6 s |
| the gate's extension (`release`) | built by `uv sync` above | 198 s |

The `release` extension gains least, because thin LTO over one codegen unit
relinks the whole extension whenever a workspace crate changes. Only the
gate builds it, once a commit; the inner loop builds `dev-wheel`.

## What `take` and `give` refuse, and why

**Each slot has its own `target/`, never a shared one.** cargo hashes a
package's path relative to the workspace root, so two checkouts building
into one `target/` overwrite each other's artifacts, and a gate can then
test another checkout's binary. There are at most four slots
(`POLARS_ONLINE_SLOTS`), each about 28 GB of builds and 0.9 GB of
environment. A slot stays warm between tasks whether or not an agent is
building in it, so there are more slots than agents running. At most two
build at once, three while the load is low, as [Keep the machine's load
down](iterate-on-a-change.md#keep-the-machines-load-down) says.

| command | refuses | because |
|---|---|---|
| `take` | a slot with uncommitted work | the work would be lost with the move |
| `take` | a branch another worktree has checked out, naming that worktree | `git checkout -B` would move the slot onto it, and a commit in either worktree would leave the other's files behind its branch |
| `give` | a slot with uncommitted work | the work would be lost with the slot |

Who holds a slot is a file in the repository's git directory, so a slot's
`git status` stays clean. `scripts/clear_caches.sh --worktrees` keeps the
slots' builds; `--slots` clears them too.

## Check what the slots share

**Check what the slots share before merging or starting a task.** A file
two branches both change costs the later one a merge of `main`, often a
conflict, and a second gate. So does a file `main` changed since a branch's
base. `scripts/slot.sh overlap` lists both cases for every taken slot, and
for any branch named after it (one about to be merged or started):

```
scripts/slot.sh overlap                   # the taken slots against each other and main
scripts/slot.sh overlap task-280-ledger   # the same, with a branch not yet in a slot
```

Branches that share a file regenerated by tooling (the state fixtures, the
golden values) or a file most changes touch (`stream.rs`, the shared test
cases) are best run one after the other. In 2026-10-25/26 four parallel
branches over `robust.rs`, `sgd.rs` and the schema-64 fixtures each paid one
to three extra gates in re-merges. Branches over separate test files merged
without one.
