<!-- kind: how-to -->
# Merge a wave of branches

This how-to is for the contributor who merges finished branches onto
`main`, and wants to land several of them behind one test gate.

**Several finished branches share one essentials gate.** A wave is the
set of finished branches merged together for that gate. Each test gate
builds and links the release extension with every core. Five at once
take 500 to 2,000 s each, against about 300 s for one alone. A branch
merged on its own also makes every other branch in flight merge `main`
and run its gate again. So each branch ends with `scripts/quick.sh
--extended` and the static checks ([Iterate on a
change](iterate-on-a-change.md)), and does not merge `main`.

**A wave of one branch skips `integration` and the essentials gate.** It
has no merges to conflict, so it merges onto `main`, a fast-forward where
it can be. The full suite, which runs before every push anyway, then runs
once on that commit (`scripts/gate.sh --extended`). The [steps](#steps) are
for a wave of two branches or more.

## Prerequisites

The steps start from branches that are finished:

- Each branch of the wave has passed `scripts/quick.sh --extended` and
  the static checks, and is committed.
- The 1-minute load under about 20 (`uptime`), as [Keep the machine's load
  down](../TESTING.md#keep-the-machines-load-down) asks before any build.

## Steps

The steps run in the main checkout:

1. **Cut an `integration` branch from `main`.**
2. **Merge the finished branches onto it in turn.**
3. **Resolve a conflict confined to generated files by regenerating them
   once.** The generated files are the state fixtures, the golden values
   and the API snapshot.
4. **Run one essentials gate, `scripts/gate.sh`, on `integration`.** Its
   last line reads `gate: PASS` or `gate: FAIL`.
5. **On PASS, move `main` to `integration`.**
6. **On a failure, find the branch.** The failure names its tests.
   `scripts/slot.sh overlap` and `git log` tie them to the branch that
   touched their files. Where that is not enough, a bisection over the
   wave's merges finds it in about log₂ of the wave's size in gates.
7. **Send that branch back to its task, and land the rest of the wave
   without it.**

To check the result, run `git log --oneline -1 main`. It names the last
merge of the wave.
