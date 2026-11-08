<!-- kind: how-to -->
# Iterate on a change

This how-to is for a contributor changing the library, who wants to run
the tests that cover a change while working on it, and the full suite
before a push.

**Run the tests for what you changed while you work, the essentials once
on each wave of merged branches, and every test before each push.** Each
level reaches what the one before it cannot, and costs more. The
essentials and the extended tier are the suite's [two
tiers](../TESTING.md#two-tiers). The times are this machine's, a 14-core
Mac shared with another user's jobs, measured on 2026-10-11 at the load
each row gives.

| when | what runs | command | time |
|---|---|---|---|
| after each change, inside a task | the tests that cover the changed paths | `scripts/quick.sh` | seconds to a minute |
| at the end of a task | the same, with the mapped files' extended tests, and the lints | `scripts/quick.sh --extended` | a minute or two |
| once on each wave of finished branches, merged together | every check, and the tests' essentials | `scripts/gate.sh` | 48 s in 8 processes at a load of 4; 111 s in 2 at a load of 20; 139 to 191 s in one |
| before every push, always | every check and every test, on the exact commit pushed | `scripts/gate.sh --extended` | 224 s in 6 processes at a load of 8; 487 to 501 s in one |

## Prerequisites

The steps need a checkout to work in and a machine with room to build:

- A checkout of the repository with its environment built (`uv sync`),
  on a branch for the change. A task works in a [worktree
  slot](use-a-worktree-slot.md), which rebuilds only what the change
  touches.
- The 1-minute load under about 20 (`uptime`), as [Keep the machine's load
  down](#keep-the-machines-load-down) asks before any build.

## Steps

The steps run from the checkout's root, in the order a task reaches them:

1. **Run `scripts/quick.sh` after each change.** It runs the tests that
   cover the paths the branch changed, and its verdict reads `PASS (the
   mapped tests only; the essentials gate runs on the merged wave)`.
2. **Run `scripts/quick.sh --dry-run` when you want to see what it would
   run.** It prints the plan without running it.
3. **At the end of the task, run `scripts/quick.sh --extended`.** It adds
   the mapped files' extended tests.
4. **Run the lints the gate runs.** These are `cargo fmt --all -- --check`,
   `cargo clippy --workspace --all-targets --locked -- -D warnings`, `uv
   run ruff format --check .`, `uv run ruff check .` and `uv run mypy`.
   Each passes with no finding.
5. **Commit the task, and leave `main` unmerged.** The branch joins the
   next wave of merges ([Merge a wave of
   branches](#merge-a-wave-of-branches)).
6. **Before every push, run `scripts/gate.sh --extended` on the exact
   commit pushed.** Its last line reads `gate: PASS (the extended tier: the
   full suite, the pre-push check)`.

To check the result, read the gate's last line: `gate: PASS` names the
tier it ran, and `gate: FAIL` names the steps that failed, with the last 25
lines of each.

## What `scripts/quick.sh` runs

**`scripts/quick.sh` runs the tests that cover the paths a branch changed,
and nothing else.** It reads the paths from `git diff` against the
merge-base with `main`, with uncommitted edits and new files included. A
Rust source change runs that crate's `cargo test` and rebuilds the
extension before pytest, which would otherwise test the old binary. A
model's source in `online-core` also runs `online-polars`' state fixture
and portability tests. A bank's frozen fixture holds every model's state,
and it moves with a model's arithmetic while every `online-core` test
passes. A Python module runs the test files named after it, and any path
runs the test files that name it. Its verdict says it covers the mapped
tests only, because a change can break a test the map does not reach.

**Inside a task, the extension is built with the `dev-wheel` profile,
which takes seconds after an edit where `release` takes minutes.** The
profile is `release` with incremental compilation and 256 codegen units,
and without thin LTO, so a rebuild redoes only what changed. `quick.sh`
builds with it, and so can `uv run maturin develop --profile dev-wheel -m
crates/online-py/Cargo.toml`.

| the extension's build | `release` | `dev-wheel` |
|---|---|---|
| from nothing | 241 s (load 10) | 130 s (load 5 to 18) |
| after a new function in `ewridge.rs` | 189 s (load 8 to 11) | 4.5 s (load 14 to 24) |
| after an edit inside `solve()`, which every model's fit calls | 189 s | 37 s (load 9 to 15) |
| the Python essentials on it, one process | 116 s | 118 to 123 s |

**A `dev-wheel` build computes the same bits as a `release` build.** Rust
fuses no floating-point operations and reorders none unless the code asks,
so neither LTO nor the number of codegen units can move a result. That was
checked rather than assumed. `scripts/release_probe.py` runs every spec
over a stream built to reach every path. It gave 297 output columns over
400 rows identical in every bit under both builds, and its 31 saved states
were identical byte for byte. The whole Python suite passed on the dev
build, the goldens and the saved-state tests among them. The gate, CI and
the shipped wheel still build `release`, because what ships is what a
push is tested on.

## Merge a wave of branches

**Several branches in flight share one essentials gate.** A gate per
branch cost twice over when branches ran side by side. Each gate builds
and links the release extension with every core, so five at once took 500
to 2,000 s each where one alone takes about 300. And every branch that
landed made the others merge `main` and gate again, most often after a
clean merge. So a task ends with `scripts/quick.sh --extended` and the
lints, and does not merge `main`.

1. **Merge the finished branches in turn onto an `integration` branch cut
   from `main`.**
2. **Resolve a conflict confined to generated files by regenerating them
   once.** The generated files are the state fixtures, the golden values
   and the API snapshot.
3. **Run one essentials gate, `scripts/gate.sh`, on `integration`.**
4. **Move `main` to `integration` only on PASS.**
5. **On a failure, find the branch.** The failure names its tests, and
   `scripts/slot.sh overlap` and `git log` tie them to the branch that
   touched their files. Where that is not enough, a bisection over the
   wave's merges finds it in about log₂ of the wave's size in gates. The
   branch goes back to its task, and the rest of the wave lands without
   it.

**A wave of one branch skips `integration` and the essentials gate.** It
has no merges to conflict, so the branch merges onto `main`, a
fast-forward where it can be. The full suite, which runs before every
push anyway, runs once on that commit. Through `integration` the same
branch paid for an essentials gate and then the full one.

## How the gate runs

**At most two gates run at once on this machine.** `scripts/gate.sh` takes
one of `POLARS_ONLINE_GATES` (default 2) tokens, directories in the
repository's common git directory holding their gate's pid. It waits for
one when all are held, prints that it waited and for how long, and starts
its timing when it has a token. A token whose gate has died is taken back.
CI runs one gate per job and takes none.

**The gate runs the tests in parallel.** cargo-nextest runs each Rust test
as a process of its own, and pytest-xdist runs the Python tests in several
processes. Both run every test that the serial run runs, with the same
results:

| tests | one process | in parallel |
|---|---|---|
| the Rust essentials | 19 s, `cargo test` | 12 s: 9.5 s under nextest and 2.6 s for the 5 doctests |
| every Rust test | 182 s | 78 to 86 s under nextest, and the doctests |
| the Python essentials | 116 to 123 s | 30 s in 8 processes (39 s in 4) |
| every Python test | 302 to 318 s | 134 s: 59 s in 8 processes, then 75 s for the 20 `serial` tests alone |

**The gate starts as many processes as the machine has cores free, from 2
to 8.** It reads that as the core count less the last minute's load
average (`scripts/workers.py`), so it does not crowd out another user's
jobs, and `GATE_WORKERS=4` overrides it. Past 8, more processes buy little
and crowd the machine: 12 took the Python essentials from 30 s to 27 s and
the load average from 23 to 51. The load average counts the gate's own
last run too, so a gate started right after another runs in fewer
processes. The gate runs `cargo test --doc` beside nextest, which runs no
doctests, and it runs `cargo test` as before where nextest is not
installed. `scripts/gate.sh --serial` runs every test in one process.

**A test that reads its process's memory runs alone, after the rest.** The
leak checks of `tests/test_ffi_memory.py` and `tests/test_gbdt.py` carry the
`serial` mark. Memory that other processes take can make macOS compress or
swap pages, which moves the resident size these tests read. The gate's
parallel run leaves them out, and a second run takes them in one process,
75 s for the 20 of them. Run that way, each still fails a planted leak:
keeping a bank's every output read 155 KB a call, and keeping every
tree-model build 1,505 KB a build, while the loop without the leak passed.
The leak checks run before the tests that read a 2 GB source. Those leave
the allocator so much freed memory that a planted leak of the tree-model
builds went unseen after them.

**A test that writes outside its own temporary directory must write a whole
file in one step.** Several processes share `.cache/`, where the downloads
land. A download is written to a file named for the writing process and
renamed into place (`tests/data.py::write_atomically`), so a reader sees no
file or the whole one. The released versions' environments, which
`scripts/compare_release.py` builds, are built under a per-process name and
renamed the same way.

## Keep the machine's load down

**At most two agents build at once on this machine (three while the load
is low), each held to six build jobs and four test processes.** Each
worktree has its own `target/` and compiles Polars there, which takes 66 s
for the Rust tests and 130 s for the `dev-wheel` extension from nothing, at
a load near 5. More agents do not finish sooner, and past a point they
bring the machine down:

| what ran at once | load on 14 cores | what happened |
|---|---|---|
| eight agents, with the other user's jobs | 31 | every build and test slowed in proportion |
| five essentials gates | 78 | gates of about 300 s took 500 to 2,000 s |
| four `cargo mutants --in-diff` runs | 88, for hours | the kernel panicked (`watchdog timeout: no checkins from watchdogd in 90 seconds`) and the Mac rebooted |
| three agents resumed straight after that reboot | | the same panic, eight hours later |

A `cargo mutants` run builds and tests every mutant, each time with every
core, so two at once already saturate the machine. So, on this machine:

- **Two agents at most, three while the load is low.** A third starts only
  while the 1-minute load is under about 20, and none starts above it (the
  user, 2026-11-07). One is the default when their work is builds and
  tests rather than reading and writing.
- **Each agent exports `CARGO_BUILD_JOBS=6`.** It runs pytest at `-n 4` at
  most, and the gate at `GATE_WORKERS=4`.
- **A local `cargo mutants --in-diff` runs alone.** It takes the lock
  directory `.claude/coord/mutants.lock` (`mkdir` it, `rmdir` it after), and
  any other waits. CI's changed-lines pass on the push checks the same lines
  on GitHub's machines, so a local run is rare.
- **Check `uptime` before starting a build.** Wait while the 1-minute load is
  above about 20.
- **After a crash, look before resuming.** Read
  `/Library/Logs/DiagnosticReports/panic-full-*` for the cause, and remove a
  stale `mutants.lock` that a killed run left.

**A compiler cache does not speed up a new worktree, so it is off unless
asked for.** sccache, turned on with `POLARS_ONLINE_SCCACHE=1` in
`scripts/env.sh`, caches each crate rustc compiles, keyed by its inputs. In
a checkout whose path it had seen, a build from nothing took 33 s for the
Rust tests and 91 s for the `release` extension, against 66 s and 241 s
without it. A new worktree has a new path, and from a new path a third of
the crates missed. A crate that includes a file its build script wrote
names that file by its absolute path, which differs per checkout. Every
crate built on such a crate misses with it, serde and Polars among them.
The misses cost more than the hits saved: 92 s and 519 s, slower than no
cache. CI never turns it on (`tests/test_dev_loop.py`).
