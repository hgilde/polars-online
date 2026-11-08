<!-- kind: how-to -->
# Update the oracle ledger

This how-to is for a contributor whose change moved this library's
numbers against a third-party oracle. It reruns the oracle grids the
change touches and records their errors in the ledger, which only
tightens.

**The oracle ledger, `oracles.toml`, holds each oracle grid's error as
measured, and every run is held to those fixed numbers.** An oracle grid
is a test that holds a model to a third-party oracle over many seeds,
orders or datasets, and carries the `oracle_grid` mark. No local test
gate runs the grids. The test gate runs a cheap guard instead, which
hashes this library's outputs on each grid's inputs. It fails when a hash
or an oracle library's version moves, on the development Mac where the
hashes were written; elsewhere it reports. [The oracle ledger](../TESTING.md#the-oracle-ledger)
describes the files, how a case is judged and why the numbers are CI's.

## Prerequisites

The steps start from a guard that has failed:

- A branch whose change makes the guard in `tests/test_oracle_ledger.py`
  fail, naming the groups whose hashes or oracle versions moved.
  `scripts/quick.sh` runs that guard for the groups a change can move.
- The project's environment (`uv sync`), and the network for any grid that
  reads a downloaded dataset.

## Steps

The steps take the change from the failed guard to a ledger CI agrees with:

1. **Rerun the groups the guard names,** with `uv run python
   scripts/oracles.py rerun`, or `uv run python scripts/oracles.py rerun
   <group>` for those you name. It runs those groups' oracle grids with
   recording on.
2. **Read its verdict.** It fails if a test fails, or if a case is past the
   ledger, has fewer values than the ledger's or was not measured.
   Otherwise it applies the run to the ledger
   (new cases, versions, and on the ledger's own platform tightened
   numbers) and writes the groups' hashes to `oracle_hashes.toml`. So a
   hash changes only after a passing run of its oracles.
3. **If a case is past the ledger, find what moved it.** If the change
   is not worth having at that error, fix the change and go back to
   step 1. If it is, go on to step 4.
4. **Write the higher number into the case's table by hand.** `rerun` and
   `apply` never write a number past the ledger. Set the case's `worst`
   or `p95` in its `[case."<key>"]` table of `oracles.toml` to the new
   value.
5. **Add a `[[raised]]` entry to `oracles.toml`.** The entry names the
   `key`, the old value as `from` and the new as `to`, the `date`, the
   `reason`, and a `changelog` text that `CHANGELOG.md` contains. A worse
   error against an oracle is a *Changed* or *Fixed* matter, so any
   section of the CHANGELOG serves.
6. **Rerun the groups,** as in step 1. This time the verdict passes and
   the groups' hashes are written.
7. **Commit `oracles.toml` and `oracle_hashes.toml` with the change.**
8. **Run `scripts/gate.sh --extended` on the commit, and push it.** The
   full suite runs on the exact commit pushed ([Iterate on a
   change](iterate-on-a-change.md), step 5).
9. **After the push, read the conclusion of `oracles.yml`.** Its last job,
   `the oracle ledger`, holds every case to the ledger on CI's platform,
   Linux x86-64.
10. **If it asks to tighten, names a new case or a moved version, or
    finds a case past the ledger, download CI's measurements,** with `gh
    run download <run-id> -n oracle-ledger-measured`. The artifact holds
    `measured.json`.
11. **Apply them,** with `uv run python scripts/oracles.py apply
    measured.json`. `apply` never writes a higher number.
12. **If CI finds a case past the ledger, write CI's value by hand.** A
    number written in step 4 from a Mac's run can be. Copy the case's
    value from `measured.json` into its table, with a `[[raised]]` entry,
    as in steps 4 and 5.
13. **Commit `oracles.toml` and push it.**

To check the result, run `uv run python scripts/oracles.py raises`. It
fails on any number in the file's history that moved away from its ideal
without a `[[raised]]` entry, and on a number dropped without a
`[[retired]]` entry. The next push's `oracles.yml` then passes.

```
uv run python scripts/oracles.py rerun            # every group whose hash or versions moved
uv run python scripts/oracles.py rerun <group>    # those named
```

**A `rerun` on another platform than the ledger's writes none of its
numbers.** The ledger holds CI's numbers, so a run on a Mac is held to
`tolerance_elsewhere` (5%) past the ledger. A gain it sees is reported,
not applied. A case new to the ledger gets its table without numbers,
which `apply` on CI's next `measured.json` fills. The hash file holds one
platform's hashes, and a `rerun` on a platform other than the one the
hashes were written on (the hash file's `[meta] platform`, the development
Mac's `darwin-arm64`) rewrites that platform. So there it refuses unless
every group is re-run, since the hashes it
kept would be the other platform's, and the guard, where they were
written, would report a moved hash instead of failing on it.

## Act on each verdict

**`scripts/oracles.py check` fails on five things, each with its remedy:**

| verdict | when | what to do |
|---|---|---|
| past the ledger | a gap more than the tolerance, 2%, past the ledger's | find what moved it; if the change is worth having at that error, write the higher number and a `[[raised]]` entry (steps 3 to 6) |
| tighten | a gap more than 2% under the ledger's | `uv run python scripts/oracles.py apply measured.json`, so the gain is kept |
| version | an oracle library at another version than the ledger's | `apply`, once the numbers held on it |
| not measured | a case of the ledger that a complete run did not record | a skip (offline), or a `record` call gone |
| fewer values | a case measured over fewer values than the ledger's `n`: its worst seed over fewer seeds is no gain, so a grid cut short cannot read as a tightening | restore the seeds; cutting them is a `[[raised]]` entry for the case's `n` |

A case new to the ledger is reported and fails nothing. `scripts/oracles.py
raises` holds the file's history. A number the history holds and the file
no longer has, from a case or group dropped or renamed, fails it until a
`[[retired]]` entry names it (a case's key or a group's), with the date
and the reason.

To add a grid of your own, see [Register an oracle grid](register-an-oracle-grid.md).
