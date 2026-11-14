<!-- kind: record -->
# How the coordinator and its workers run

Moved verbatim from CLAUDE.md on 2026-11-14 (the user: cut the size of the
docs read most), so every session loads a short summary and this file holds
the detail and the reason for each rule. CLAUDE.md's "Workflow" section is
the summary.

Iterating (task 245; docs/TESTING.md, "Iterating" and "Caches and when to
clear them"): workers run `scripts/quick.sh` while iterating and finish
with `scripts/quick.sh --extended` plus the lints, not a full gate, and
never merge `main` unless asked. The coordinator merges each wave of
finished branches onto an `integration` branch cut from `main`, regenerates
any generated files that conflict (fixtures, goldens, the API snapshot)
once, runs ONE essentials gate, and fast-forwards `main` on PASS; a failure
is mapped to a branch by the files it touched (`scripts/slot.sh overlap`)
or bisected over the wave's merges (the user, 2026-10-26: parallel gates
and the re-gates of conflicting merges were the agents' two bottlenecks).
A wave of one branch skips `integration` and the essentials gate: it merges
onto `main` (a fast-forward where it can be) and the full tier runs once, on
that sha, before the push (the user, 2026-11-03: "why bother with an
integration branch").
`scripts/gate.sh` takes one of two machine-wide tokens and waits for one
(`POLARS_ONLINE_GATES`). The full tier runs, always, before every push on
the exact sha.
Run at most two building agents at once on this machine -- three while the
1-minute load stays under about 20 (the user, 2026-11-07) -- each with
`CARGO_BUILD_JOBS=6`, pytest at `-n 4` and the gate at `GATE_WORKERS=4`;
a local `cargo mutants` runs alone, under `.claude/coord/mutants.lock`;
check `uptime` first and wait above a load of about 20 (docs/TESTING.md,
"At most two agents": four parallel mutants runs at a load of 88 panicked
the Mac twice, 2026-10-27/28). Coordination files -- briefs, the queue,
the worker rules, survivor lists -- live in `.claude/coord/` (locally
excluded), never under `/tmp`, which a restart empties. A worker gets a
worktree slot, not a fresh worktree (task 248; docs/TESTING.md, "Worktree
slots"): the coordinator runs `scripts/slot.sh take <branch> main`, starts
the worker without `isolation`, told to work only in that path, and runs
`scripts/slot.sh give <n>` after merging; a warm slot rebuilds the tests in
23 s and the dev wheel in 6 s where a fresh worktree takes 79 s and 145 s. Nobody runs a full
`cargo mutants` pass locally (the user, 2026-10-11: "Every task seems to
spend a lot of time waiting for mutants", then "we need a thorough mutants
check to run the smallest number of times"): the one thorough check is CI's
changed-lines pass on the push, sharded across runners and free; its
survivors are a follow-up batch, and a fix is checked locally only with an
`--in-diff` of the fix itself (`--copy-target=true --test-tool nextest`).
So push each merged, gated wave as it lands (the user, 2026-10-12: "Yes"
to it): a push needs the full gate on the exact sha and no open bug, and
then no further word; small pushes keep each CI mutants pass short and its
survivors fresh. Dispatching a release still needs the user's explicit
word, and its PyPI approval is always the user's click. Workers lean on activation tests instead. After each push
wave and its branch and worktree cleanup, run `scripts/clear_caches.sh
--worktrees` (or whenever free disk is under 100 GB). sccache stays off:
from a new worktree path a third of the crates miss its cache, so it is
slower there; `POLARS_ONLINE_SCCACHE=1` turns it on for one checkout.
